//! Compile a model straight from its path, without reading it into a `Model` first.
mod fixtures;

use fixtures::alexnet as fixture;
use openvino::{Core, DeviceType, ElementType, Shape, Tensor};

#[test]
fn compile_model_from_file_matches_read_then_compile() -> anyhow::Result<()> {
    let mut core = Core::new()?;
    let graph = fixture::graph().to_string_lossy().into_owned();
    let weights = fixture::weights().to_string_lossy().into_owned();

    let mut from_file = core.compile_model_from_file(&graph, DeviceType::CPU)?;
    let model = core.read_model_from_file(&graph, &weights)?;
    let mut from_model = core.compile_model(&model, DeviceType::CPU)?;

    let shape = Shape::new(&[1, 3, 227, 227])?;
    let mut input = Tensor::new(ElementType::F32, &shape)?;
    for (i, x) in input.get_data_mut::<f32>()?.iter_mut().enumerate() {
        *x = (i % 251) as f32 / 251.0;
    }
    let mut outputs = Vec::new();
    for compiled in [&mut from_file, &mut from_model] {
        let mut request = compiled.create_infer_request()?;
        request.set_tensor("data", &input)?;
        request.infer()?;
        outputs.push(request.get_tensor("prob")?.get_data::<f32>()?.to_vec());
    }
    assert_eq!(outputs[0], outputs[1]);
    Ok(())
}

#[test]
fn compile_model_from_file_reports_a_missing_file() {
    let mut core = Core::new().unwrap();
    assert!(core
        .compile_model_from_file("/nonexistent/model.xml", DeviceType::CPU)
        .is_err());
}
