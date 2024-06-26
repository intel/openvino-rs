mod fixtures;
mod util;

use fixtures::mobilenet::Fixture;
use openvino::{Core, DeviceType, ElementType, Shape, Tensor};
use std::fs;

#[test]
fn classify_mobilenet() -> anyhow::Result<()> {
    let mut core = Core::new()?;
    // let model = core.read_model_from_file(
    //     &Fixture::graph().to_string_lossy(),
    //     &Fixture::weights().to_string_lossy(),
    // )?;
    let xml = fs::read_to_string(Fixture::graph())?;
    let weights = fs::read(Fixture::weights())?;

    //Construct new tensor with data.
    let dims: [i64; 2] = [1, weights.len() as i64];
    let shape = Shape::new(&dims)?;
    let mut weights_tensor = Tensor::new(ElementType::U8, &shape)?;

    let buffer = weights_tensor.buffer_mut()?;
    for (index, bytes) in weights.iter().enumerate() {
        buffer[index] = *bytes;
    }

    // let buffer = weights_tensor.buffer_mut()?;
    // buffer.copy_from_slice(&weights);
    let model = core.read_model_from_buffer(xml.as_bytes(), Some(&weights_tensor))?;

    //std::mem::drop(weights_tensor);

    // Compile the model and infer the results.
    //let executable_model = core.compile_model(&model, DeviceType::CPU)?;
    //println!("{}",executable_model.get_input()?.get_name()?);
    Ok(())
}
