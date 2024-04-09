//! Demonstrates using `openvino-rs` to classify an image using an Inception SSD model and a prepared input tensor. See
//! [README](fixtures/inception/README.md) for details on how this test fixture was prepared.
mod fixtures;
mod util;

use anyhow::Ok;
use fixtures::inception::Fixture;
use openvino::{Core, ElementType, Layout, PrePostProcess, Shape, Tensor};
use std::fs;
use util::{Prediction, Predictions};

#[test]
fn classify_inception() -> anyhow::Result<()> {
    //initialize openvino runtime core
    let mut core = Core::new()?;

    //Read the model
    let mut model = core.read_model_from_file(
        &Fixture::graph().to_string_lossy(),
        &Fixture::weights().to_string_lossy(),
    )?;

    //Set up input
    let data = fs::read(Fixture::tensor())?;
    let input_shape = Shape::new(&vec![1, 299, 299, 3])?;
    let element_type = ElementType::F32;
    let tensor = Tensor::new_from_host_ptr(element_type, &input_shape, &data)?;

    let pre_post_process = PrePostProcess::new(&mut model)?;
    let input_info = pre_post_process.get_input_info_by_name("input")?;
    let mut input_tensor_info = input_info.preprocess_input_info_get_tensor_info()?;
    input_tensor_info.preprocess_input_tensor_set_from(&tensor)?;

    let input_layout = Layout::new("NHWC")?;
    input_tensor_info.preprocess_input_tensor_set_layout(&input_layout)?;
    let mut preprocess_steps = input_info.get_preprocess_steps()?;
    preprocess_steps.preprocess_steps_resize(0)?;

    let model_info = input_info.get_model_info()?;
    let model_layout = Layout::new("NCHW")?;
    model_info.model_info_set_layout(&model_layout)?;

    let new_model = pre_post_process.build_new_model()?;

    let input_port = model.get_input_by_index(0)?;
    assert_eq!(input_port.get_name()?, "input");

    //Set up output
    let output_port = model.get_output_by_index(0)?;
    assert_eq!(output_port.get_name()?, "InceptionV3/Predictions/Softmax");

    // Load the model.
    let mut executable_model = core.compile_model(&new_model, "CPU")?;
    let mut infer_request = executable_model.create_infer_request()?;

    // Execute inference.
    infer_request.set_tensor("input", &tensor)?;
    infer_request.infer()?;
    let mut results = infer_request.get_tensor(&output_port.get_name()?)?;

    let buffer = results.get_data::<f32>()?.to_vec();

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

    Ok(())
}
