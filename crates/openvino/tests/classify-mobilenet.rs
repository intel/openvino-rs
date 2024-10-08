//! Demonstrates using `openvino` to classify an image using an MobileNet model and a prepared input
//! tensor.

mod fixtures;
mod util;

use fixtures::mobilenet as fixture;
use openvino::{
    prepostprocess, Core, DeviceType, ElementType, Layout, ResizeAlgorithm, Shape, Tensor,
};
use std::fs;
use util::{Prediction, Predictions};

#[test]
fn classify_mobilenet() -> anyhow::Result<()> {
    let mut core = Core::new()?;
    let mut model = core.read_model_from_file(
        &fixture::graph().to_string_lossy(),
        &fixture::weights().to_string_lossy(),
    )?;

    let output_port = model.get_output_by_index(0)?;
    assert_eq!(output_port.get_name()?, "MobilenetV2/Predictions/Reshape_1");
    assert_eq!(model.get_input_by_index(0)?.get_name()?, "input");

    // Retrieve the tensor from the test fixtures.
    let data = fs::read(fixture::tensor())?;
    let input_shape = Shape::new(&[1, 224, 224, 3])?;
    let element_type = ElementType::F32;
    let mut tensor = Tensor::new(element_type, &input_shape)?;
    let buffer = tensor.get_raw_data_mut()?;
    buffer.copy_from_slice(&data);

    // Pre-process the input by:
    // - converting NHWC to NCHW
    // - resizing the input image
    let pre_post_process = prepostprocess::Pipeline::new(&mut model)?;
    let input_info = pre_post_process.get_input_info_by_name("input")?;
    let mut input_tensor_info = input_info.get_tensor_info()?;
    input_tensor_info.set_from(&tensor)?;
    input_tensor_info.set_layout(Layout::new("NHWC")?)?;
    let mut steps = input_info.get_steps()?;
    steps.resize(ResizeAlgorithm::Linear)?;
    let mut model_info = input_info.get_model_info()?;
    model_info.set_layout(Layout::new("NCHW")?)?;
    let output_info = pre_post_process.get_output_info_by_index(0)?;
    let mut output_tensor_info = output_info.get_tensor_info()?;
    output_tensor_info.set_element_type(ElementType::F32)?;
    let new_model = pre_post_process.build_new_model()?;

    // Compile the model and infer the results.
    let mut executable_model = core.compile_model(&new_model, DeviceType::CPU)?;
    let mut infer_request = executable_model.create_infer_request()?;
    infer_request.set_tensor("input", &tensor)?;
    infer_request.infer()?;
    let results = infer_request.get_tensor(&output_port.get_name()?)?;

    // Sort results. It is unclear why the MobileNet output indices are "off by one" but the
    // `.skip(1)` below seems necessary to get results that make sense (e.g. 763 = "revolver" vs 762
    // = "restaurant").
    let buffer = results.get_data::<f32>()?.to_vec();
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
