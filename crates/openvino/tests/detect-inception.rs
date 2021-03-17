//! Demonstrates using `openvino-rs` to classify an image using an Inception SSD model and a prepared input tensor. See
//! [README](fixtures/inception/README.md) for details on how this test fixture was prepared.
mod fixtures;

use fixtures::inception_ssd::Fixture;
use openvino::{Blob, Core, Layout, Precision, ResizeAlgorithm, TensorDesc};
use std::fs;

#[test]
fn detect_inception() {
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

    // Read the image.
    let tensor_data = fs::read(Fixture::tensor()).unwrap();
    let tensor_desc = TensorDesc::new(Layout::NHWC, &[1, 3, 481, 640], Precision::U8);
    let blob = Blob::new(tensor_desc, &tensor_data).unwrap();

    // Execute inference.
    infer_request.set_blob(input_name, blob).unwrap();
    infer_request.infer().unwrap();
    let mut results = infer_request.get_blob(output_name).unwrap();
    let buffer = unsafe { results.buffer_mut_as_type::<f32>().unwrap().to_vec() };

    // Sort results (TODO extract bounding boxes instead).
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

    // This above results should match the output of running OpenVINO's
    // `object_detection_sample_ssd` with the same inputs. This test incorrectly uses result
    // sorting instead of extracting the bounding boxes like `object_detection_sample_ssd` does
    // (FIXME):
    // $ bin/intel64/Debug/object_detection_sample_ssd -m ../inception-ssd.xml -i ../pizza.jpg
    // [ INFO ] InferenceEngine:
    //     API version ............ 2.1
    //     Build .................. custom_master_a1d858c5028c1a26d37286913d64028849454b75
    //     Description ....... API
    // Parsing input parameters
    // [ INFO ] Files were added: 1
    // [ INFO ]     ../pizza.jpg
    // [ INFO ] Loading Inference Engine
    // [ INFO ] Device info:
    //     CPU
    //     MKLDNNPlugin version ......... 2.1
    //     Build ........... custom_master_a1d858c5028c1a26d37286913d64028849454b75
    // [ INFO ] Loading network files:
    //     ../inception-ssd.xml
    // [ INFO ] Preparing input blobs
    // [ INFO ] Batch size is 1
    // [ INFO ] Preparing output blobs
    // [ INFO ] Loading model to the device
    // [ INFO ] Create infer request
    // [ WARNING ] Image is resized from (640, 481) to (300, 300)
    // [ INFO ] Batch size is 1
    // [ INFO ] Start inference
    // [ INFO ] Processing output blobs
    // [0,1] element, prob = 0.975312    (1,19)-(270,389) batch id : 0 WILL BE PRINTED!
    // [1,1] element, prob = 0.953244    (368,17)-(640,393) batch id : 0 WILL BE PRINTED!
    // [2,59] element, prob = 0.993812    (143,280)-(502,423) batch id : 0 WILL BE PRINTED!
    // [3,67] element, prob = 0.301402    (5,369)-(582,480) batch id : 0
    // [ INFO ] Image out_0.bmp created!
    // [ INFO ] Execution successful
}

/// A structure for holding the `(category, probability)` pair extracted from the output tensor of
/// the OpenVINO classification.
#[derive(Debug, PartialEq)]
struct Result(usize, f32);
type Results = Vec<Result>;
