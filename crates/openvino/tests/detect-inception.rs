//! Demonstrates using `openvino-rs` to classify an image using an Inception SSD model and a prepared input tensor. See
//! [README](fixtures/inception/README.md) for details on how this test fixture was prepared.
mod fixtures;
//mod util;

use fixtures::inception_ssd::Fixture;
use openvino::{Core, ElementType, Layout, Model, PrePostprocess, Shape, Tensor};
use std::fs;

#[test]
fn detect_inception() {
    //create an emtpy model for preprocess build
    let mut new_model = Model::new().unwrap();

    //initialize openvino runtime core
    let mut core = Core::new().unwrap();

    //Read the model
    let model = core
        .read_model_from_file(
            &Fixture::graph().to_string_lossy(),
            &Fixture::weights().to_string_lossy(),
        )
        .unwrap();

    //Set up output
    let output_port = model.get_output_by_index(0).unwrap();
    assert_eq!(output_port.get_name().unwrap(), "DetectionOutput");

    let input_port = model.get_input_by_index(0).unwrap();
    assert_eq!(input_port.get_name().unwrap(), "image_tensor");

    //Set up input
    let data = fs::read(Fixture::tensor()).unwrap();
    let input_shape = Shape::new(&vec![1, 481, 640, 3]).unwrap();
    let element_type = ElementType::U8;
    let tensor = Tensor::new_from_host_ptr(element_type, input_shape, &data).unwrap();
    let pre_post_process = PrePostprocess::new(&model).unwrap();
    let input_info = pre_post_process
        .get_input_info_by_name("image_tensor")
        .unwrap();
    let mut input_tensor_info = input_info.preprocess_input_info_get_tensor_info().unwrap();
    input_tensor_info
        .preprocess_input_tensor_set_from(&tensor)
        .unwrap();

    let layout_tensor_string = "NHWC";
    let input_layout = Layout::new(&layout_tensor_string).unwrap();
    input_tensor_info
        .preprocess_input_tensor_set_layout(&input_layout)
        .unwrap();
    let mut preprocess_steps = input_info.get_preprocess_steps().unwrap();
    preprocess_steps.preprocess_steps_resize(0).unwrap();
    preprocess_steps
        .preprocess_convert_element_type(ElementType::F32)
        .unwrap();
    //Layout conversion is supposed to be implicit, but can be done explicitly like shown below in comments
    // let input_layout_convert = Layout::new("NCHW");
    // preprocess_steps.preprocess_convert_layout(input_layout_convert);

    let model_info = input_info.get_model_info().unwrap();
    let layout_string = "NCHW";
    let model_layout = Layout::new(&layout_string).unwrap();
    model_info.model_info_set_layout(model_layout).unwrap();

    let output_info = pre_post_process.get_output_info_by_index(0).unwrap();
    let output_tensor_info = output_info.get_output_info_get_tensor_info().unwrap();
    output_tensor_info
        .preprocess_set_element_type(ElementType::F32)
        .unwrap();

    pre_post_process.build(&mut new_model).unwrap();

    // Load the model.
    let mut executable_model = core.compile_model(new_model, "CPU").unwrap();
    let mut infer_request = executable_model.create_infer_request().unwrap();

    // Execute inference.
    infer_request.set_tensor("image_tensor", &tensor).unwrap();
    infer_request.infer().unwrap();
    let mut results = infer_request
        .get_tensor(&output_port.get_name().unwrap())
        .unwrap();

    let buffer = results.get_data::<f32>().unwrap().to_vec();

    // Sort results (TODO extract bounding boxes instead).
    let mut results: Results = buffer
        .iter()
        .enumerate()
        .map(|(c, p)| Result(c, *p))
        .collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    //Result buffer seem off by 1
    assert_eq!(
        &results[1..5],
        &[
            Result(15, 59.0),
            Result(1, 1.0),
            Result(8, 1.0),
            Result(12, 1.0),
            //Result(16, 0.9939936),
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
