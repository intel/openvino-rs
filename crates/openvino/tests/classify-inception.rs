//! Demonstrates using `openvino-rs` to classify an image using an Inception SSD model and a prepared input tensor. See
//! [README](fixtures/inception/README.md) for details on how this test fixture was prepared.
mod fixtures;
mod util;

use fixtures::inception::Fixture;
use openvino::{Blob, Core, Layout, Precision, TensorDesc};
use std::fs;
use util::{Prediction, Predictions};

#[test]
fn classify_inception() {
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
    assert_eq!(output_name, "InceptionV3/Predictions/Softmax");

    // Load the network.
    let mut executable_network = core.load_network(&network, "CPU").unwrap();
    let mut infer_request = executable_network.create_infer_request().unwrap();

    // Read the image.
    let tensor_data = fs::read(Fixture::tensor()).unwrap();
    let tensor_desc = TensorDesc::new(Layout::NHWC, &[1, 3, 299, 299], Precision::FP32);
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

    // Note that these results appear to be off-by-one: pizza should be ID 963.
    results[0].assert_approx_eq((964, 0.9648312));
    results[1].assert_approx_eq((763, 0.0015633557));
    results[2].assert_approx_eq((412, 0.0007776478));
    results[3].assert_approx_eq((814, 0.0006391522));
    results[4].assert_approx_eq((924, 0.0006150733));

    // The results above almost match the output of OpenVINO's `hello_classification` with similar
    // inputs:
    // $ bin/intel64/Debug/hello_classification ../inception.xml ../pizza.jpg CPU
    // Top 10 results:
    // Image ../pizza.jpg
    // classid probability
    // ------- -----------
    // 964     0.9656160
    // 763     0.0015505
    // 412     0.0007806
    // 924     0.0006135
    // 814     0.0006102
    // 966     0.0005903
    // 960     0.0004972
    // 522     0.0003951
    // 927     0.0003644
    // 923     0.0002908
}
