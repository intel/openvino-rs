//! Demonstrates using `openvino-rs` to classify an image using an MobileNet model and a prepared
//! input tensor. See [README](fixtures/inception/README.md) for details on how this test fixture
//! was prepared.
mod fixtures;
mod util;

use fixtures::mobilenet::Fixture;
use openvino::{Core, ElementType, Layout, PrePostProcess, Shape, Tensor};
use std::fs;
use util::{Prediction, Predictions};

#[test]
fn classify_mobilenet() -> anyhow::Result<()> {
    //initialize openvino runtime core
    let mut core = Core::new()?;

    //Read the model
    let mut model = core.read_model_from_file(Fixture::graph(), Fixture::weights())?;

    //Set up output port of model
    let output_port = model.output_by_index(0)?;
    assert_eq!(output_port.name()?, "MobilenetV2/Predictions/Reshape_1");

    //Set up input port of model
    let input_port = model.input_by_index(0)?;
    assert_eq!(input_port.name()?, "input");

    //Set up input
    let data = fs::read(Fixture::tensor())?;
    let input_shape = Shape::new(&vec![1, 224, 224, 3])?;
    let element_type = ElementType::F32;
    let tensor = Tensor::new_from_host_ptr(element_type, &input_shape, &data)?;

    //configure preprocessing
    let pre_post_process = PrePostProcess::new(&mut model)?;
    let input_info = pre_post_process.input_info_by_name("input")?;
    let mut input_tensor_info = input_info.tensor_info()?;
    input_tensor_info.set_from(&tensor)?;

    //set layout of input tensor
    let layout_tensor_string = "NHWC";
    let input_layout = Layout::new(&layout_tensor_string)?;
    input_tensor_info.set_layout(&input_layout)?;

    //set any preprocessing steps
    let mut preprocess_steps = input_info.preprocess_steps()?;
    preprocess_steps.resize(0)?;
    let model_info = input_info.model_info()?;

    //set model input layout
    let layout_string = "NCHW";
    let model_layout = Layout::new(&layout_string)?;
    model_info.set_layout(&model_layout)?;

    let output_info = pre_post_process.output_info_by_index(0)?;
    let output_tensor_info = output_info.tensor_info()?;
    output_tensor_info.set_element_type(ElementType::F32)?;

    let new_model = pre_post_process.build()?;

    // Load the model.
    let mut executable_model = core.compile_model(&new_model, "CPU")?;

    //create an inference request
    let mut infer_request = executable_model.create_infer_request()?;

    //Prepare input
    infer_request.set_tensor("input", &tensor)?;

    // Execute inference.
    infer_request.infer()?;
    let mut results = infer_request.tensor(&output_port.name()?)?;

    let buffer = results.data::<f32>()?.to_vec();
    // Sort results. It is unclear why the MobileNet output indices are "off by one" but the
    // `.skip(1)` below seems necessary to get results that make sense (e.g. 763 = "revolver" vs 762
    // = "restaurant").
    let mut results: Predictions = buffer
        .iter()
        .skip(1)
        .enumerate()
        .map(|(c, p)| Prediction::new(c, *p))
        .collect();
    results.sort();

    // Compare results using approximate FP comparisons; annotated with classification tags from
    // https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a.
    results[0].assert_approx_eq((963, 0.7134405)); // pizza
    results[1].assert_approx_eq((762, 0.0715866)); // restaurant
    results[2].assert_approx_eq((909, 0.0360171)); // wok
    results[3].assert_approx_eq((926, 0.0160412)); // hot pot
    results[4].assert_approx_eq((567, 0.0152781)); // frying pan

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

    Ok(())
}
