//! Demonstrates using `openvino-rs` to classify an image using an MobileNet model and a prepared
//! input tensor. See [README](fixtures/inception/README.md) for details on how this test fixture
//! was prepared.
mod fixtures;

use fixtures::mobilenet::Fixture;
use float_cmp::approx_eq;
use openvino::{Blob, Core, Layout, Precision, TensorDesc};
use std::fs;

#[test]
fn classify_mobilenet() {
    let mut core = Core::new(None).unwrap();
    let network = core
        .read_network_from_file(
            &Fixture::graph().to_string_lossy(),
            &Fixture::weights().to_string_lossy(),
        )
        .unwrap();

    let input_name = &network.get_input_name(0).unwrap();
    assert_eq!(input_name, "input");
    let output_name = &network.get_output_name(0).unwrap();
    assert_eq!(output_name, "MobilenetV2/Predictions/Reshape_1");

    // Load the network.
    let mut executable_network = core.load_network(&network, "CPU").unwrap();
    let mut infer_request = executable_network.create_infer_request().unwrap();

    // Read the image.
    let tensor_data = fs::read(Fixture::tensor()).unwrap();
    let tensor_desc = TensorDesc::new(Layout::NHWC, &[1, 3, 224, 224], Precision::FP32);
    let blob = Blob::new(tensor_desc, &tensor_data).unwrap();

    // Execute inference.
    infer_request.set_blob(input_name, blob).unwrap();
    infer_request.infer().unwrap();
    let mut results = infer_request.get_blob(output_name).unwrap();
    let buffer = unsafe { results.buffer_mut_as_type::<f32>().unwrap().to_vec() };

    // Sort results. It is unclear why the MobileNet output indices are "off by one" but the
    // `.skip(1)` below seems necessary to get results that make sense (e.g. 763 = "revolver" vs 762
    // = "restaurant").
    let mut results: Results = buffer
        .iter()
        .skip(1)
        .enumerate()
        .map(|(c, p)| Result(c, *p))
        .collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Compare results using approximate FP comparisons; annotated with classification tag from
    // https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a.
    results[0].assert_approx_eq(&Result(963, 0.7134405)); // pizza
    results[1].assert_approx_eq(&Result(762, 0.0715866)); // restaurant
    results[2].assert_approx_eq(&Result(909, 0.0360171)); // wok
    results[3].assert_approx_eq(&Result(926, 0.0160412)); // hot pot
    results[4].assert_approx_eq(&Result(567, 0.0152781)); // frying pan

    // This above results almost match (see "off by one" comment above) the output of running
    // OpenVINO's `hello_classification` with the same inputs:
    // $ bin/intel64/Debug/hello_classification /tmp/mobilenet.xml /tmp/val2017/000000062808.jpg CPU
    // Image /tmp/val2017/000000062808.jpg
    // classid probability
    // ------- -----------
    // 964     0.7134405
    // 763     0.0715866
    // 910     0.0360171
    // 927     0.0160412
    // 568     0.0152781
    // 924     0.0148565
    // 500     0.0093886
    // 468     0.0073142
    // 965     0.0058377
    // 545     0.0043731
}

/// A structure for holding the `(category, probability)` pair extracted from the output tensor of
/// the OpenVINO classification.
#[derive(Debug, PartialEq)]
struct Result(usize, f32);
type Results = Vec<Result>;

impl Result {
    fn assert_approx_eq(&self, expected: &Result) {
        assert_eq!(
            self.0, expected.0,
            "Expected class ID {} but found {}",
            expected.0, self.0
        );
        let approx_matches = approx_eq!(f32, self.1, expected.1, ulps = 2, epsilon = 0.01);
        assert!(
            approx_matches,
            "Expected probability {} but found {} (outside of tolerance)",
            expected.1, self.1
        );
    }
}
