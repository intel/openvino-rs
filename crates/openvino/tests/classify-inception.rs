//! Demonstrates using `openvino-rs` to classify an image using an Inception SSD model and a prepared input tensor. See
//! [README](fixtures/inception/README.md) for details on how this test fixture was prepared.
mod fixtures;

use fixtures::inception::Fixture;
use openvino::{Blob, Core, Layout, Precision, ResizeAlgorithm, TensorDesc};
use std::fs;

#[test]
fn classify_inception() {
    let mut core = Core::new(None).unwrap();
    let mut network = core
        .read_network_from_file(
            &Fixture::graph().to_string_lossy(),
            &Fixture::weights().to_string_lossy(),
        )
        .unwrap();

    let input_name = &network.get_input_name(0).unwrap();
    assert_eq!(input_name, "image_tensor");
    let output_name = &network.get_output_name(0).unwrap();
    assert_eq!(output_name, "DetectionOutput");

    // Prepare inputs and outputs for resizing, since our input tensor is not the size the model expects.
    network
        .set_input_resize_algorithm(input_name, ResizeAlgorithm::RESIZE_BILINEAR)
        .unwrap();
    network.set_input_layout(input_name, Layout::NHWC).unwrap();
    network
        .set_input_precision(input_name, Precision::U8)
        .unwrap();
    network
        .set_output_precision(output_name, Precision::FP32)
        .unwrap();

    // Load the network.
    let mut executable_network = core.load_network(&network, "CPU").unwrap();
    let mut infer_request = executable_network.create_infer_request().unwrap();
    // TODO eventually, this should not panic: infer_request.set_batch_size(1).unwrap();

    // Read the image.
    let tensor_data = fs::read(Fixture::tensor()).unwrap();
    let tensor_desc = TensorDesc::new(Layout::NHWC, &[1, 3, 481, 640], Precision::U8);
    let blob = Blob::new(tensor_desc, &tensor_data).unwrap();

    // Execute inference.
    infer_request.set_blob(input_name, blob).unwrap();
    infer_request.infer().unwrap();
    let mut results = infer_request.get_blob(output_name).unwrap();
    let buffer = unsafe { results.buffer_mut_as_type::<f32>().unwrap().to_vec() };

    // Sort results.
    let mut results: Results = buffer
        .iter()
        .enumerate()
        .map(|(c, p)| Result(c, *p))
        .collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    assert_eq!(
        &results[..5],
        &[
            Result(15, 59.0),
            Result(1, 1.0),
            Result(8, 1.0),
            Result(12, 1.0),
            Result(16, 0.9939936),
        ][..]
    )

    // This above results match the output of running OpenVINO's `hello_classification` with the same inputs:
    // $ bin/intel64/Debug/hello_classification /tmp/fixture/frozen_inference_graph.xml /tmp/fixture/000000062808.jpg CPU
    // Top 10 results:
    // Image /tmp/fixture/000000062808.jpg
    // classid probability
    // ------- -----------
    // 15      59.0000000
    // 1       1.0000000
    // 12      1.0000000
    // 8       1.0000000
    // 16      0.9939936
    // 2       0.9750488
    // 9       0.9535966
    // 20      0.8796915
    // 13      0.8178773
    // 6       0.8092338
}

/// A structure for holding the `(category, probability)` pair extracted from the output tensor of
/// the OpenVINO classification.
#[derive(Debug, PartialEq)]
struct Result(usize, f32);
type Results = Vec<Result>;
