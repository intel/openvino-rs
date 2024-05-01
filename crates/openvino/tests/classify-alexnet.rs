//! Demonstrates using `openvino-rs` to classify an image using an AlexNet model and a prepared input tensor. See
//! [README](fixtures/alexnet/README.md) for details on how this test fixture was prepared.
mod fixtures;
mod util;

use anyhow::Ok;
use fixtures::alexnet::Fixture;
use openvino::{prepostprocess, Core, ElementType, Layout, Shape, Tensor};
use std::fs;
use util::{Prediction, Predictions};

#[test]
fn classify_alexnet() -> anyhow::Result<()> {
    let mut core = Core::new()?;
    let mut model = core.read_model_from_file(
        &Fixture::graph().to_string_lossy(),
        &Fixture::weights().to_string_lossy(),
    )?;

    let output_port = model.get_output_by_index(0)?;
    assert_eq!(output_port.get_name()?, "prob");
    assert_eq!(model.get_input_by_index(0)?.get_name()?, "data");

    // Retrieve the tensor from the test fixtures.
    let data = fs::read(Fixture::tensor())?;
    let input_shape = Shape::new(&vec![1, 227, 227, 3])?;
    let element_type = ElementType::F32;
    let tensor = Tensor::new_from_host_ptr(element_type, &input_shape, &data)?;

    // Pre-process the input by:
    // - converting NHWC to NCHW
    // - resizing the input image
    let pre_post_process = prepostprocess::PrePostProcess::new(&mut model)?;
    let input_info = pre_post_process.get_input_info_by_name("data")?;
    let mut input_tensor_info = input_info.preprocess_input_info_get_tensor_info()?;
    input_tensor_info.preprocess_input_tensor_set_from(&tensor)?;
    input_tensor_info.preprocess_input_tensor_set_layout(&Layout::new("NHWC")?)?;
    let mut preprocess_steps = input_info.get_preprocess_steps()?;
    preprocess_steps.preprocess_steps_resize(0)?;
    let model_info = input_info.get_model_info()?;
    model_info.model_info_set_layout(&Layout::new("NCHW")?)?;
    let output_info = pre_post_process.get_output_info_by_index(0)?;
    let output_tensor_info = output_info.get_output_info_get_tensor_info()?;
    output_tensor_info.preprocess_set_element_type(ElementType::F32)?;
    let new_model = pre_post_process.build_new_model()?;

    // Compile the model and infer the results.
    let mut executable_model = core.compile_model(&new_model, "CPU")?;
    let mut infer_request = executable_model.create_infer_request()?;
    infer_request.set_tensor("data", &tensor)?;
    infer_request.infer()?;
    let mut results = infer_request.get_tensor(&output_port.get_name()?)?;

    // Sort results.
    let buffer = results.get_data::<f32>()?.to_vec();
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

    Ok(())
}
