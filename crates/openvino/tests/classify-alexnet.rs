//! Demonstrates using `openvino-rs` to classify an image using an AlexNet model and a prepared input tensor. See
//! [README](fixtures/alexnet/README.md) for details on how this test fixture was prepared.
mod fixtures;
mod util;

use fixtures::alexnet::Fixture;
use openvino::{Blob, Core, Layout, Precision, TensorDesc};
use std::fs;
use util::{Prediction, Predictions};

#[test]
fn classify_alexnet() {
    let mut core = Core::new(None).unwrap();
    let mut network = core
        .read_network_from_file(
            &Fixture::graph().to_string_lossy(),
            &Fixture::weights().to_string_lossy(),
        )
        .unwrap();

    let input_name = &network.get_input_name(0).unwrap();
    assert_eq!(input_name, "data");
    network.set_input_layout(input_name, Layout::NHWC).unwrap();
    let output_name = &network.get_output_name(0).unwrap();
    assert_eq!(output_name, "prob");

    // Load the network.
    let mut executable_network = core.load_network(&network, "CPU").unwrap();
    let mut infer_request = executable_network.create_infer_request().unwrap();

    // Read the image.
    let tensor_data = fs::read(Fixture::tensor()).unwrap();
    let tensor_desc = TensorDesc::new(Layout::NHWC, &[1, 3, 227, 227], Precision::FP32);
    let blob = Blob::new(tensor_desc, &tensor_data).unwrap();

    // Execute inference.
    infer_request.set_blob(input_name, blob).unwrap();
    infer_request.infer().unwrap();
    let mut results = infer_request.get_blob(output_name).unwrap();
    let buffer = unsafe { results.buffer_mut_as_type::<f32>().unwrap().to_vec() };

    // Sort results.
    let mut results: Predictions = buffer
        .iter()
        .enumerate()
        .map(|(c, p)| Prediction::new(c, *p))
        .collect();
    results.sort();

    // Compare results using approximate FP comparisons; annotated with classification tags from
    // https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a.
    results[0].assert_approx_eq((963, 0.5321184)); // pizza
    results[1].assert_approx_eq((923, 0.1050855)); // plate
    results[2].assert_approx_eq((926, 0.1022315)); // hot pot
    results[3].assert_approx_eq((909, 0.0614674)); // wok
    results[4].assert_approx_eq((762, 0.0549604)); // restaurant

    // This above results match the output of running OpenVINO's `hello_classification` with the same inputs:
    // $ bin/intel64/Debug/hello_classification /tmp/alexnet/bvlc_alexnet.xml /tmp/alexnet/val2017/000000062808.jpg CPU
    // Top 10 results:
    // Image /tmp/alexnet/val2017/000000062808.jpg
    // classid probability
    // ------- -----------
    // 963     0.5321184
    // 923     0.1050855
    // 926     0.1022315
    // 909     0.0614674
    // 762     0.0549604
    // 959     0.0284412
    // 962     0.0149365
    // 118     0.0143028
    // 935     0.0130160
    // 965     0.0094148
}
