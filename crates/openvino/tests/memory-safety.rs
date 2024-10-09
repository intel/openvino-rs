//! This test originates from some strange segfaults that we observed while using the OpenVINO C
//! library. Because OpenVINO is taking a pointer to a tensor when constructing a model, we want to
//! be sure that we do the right thing on this side of the FFI boundary.

mod fixtures;
mod util;

use fixtures::mobilenet as fixture;
use openvino::{Core, DeviceType, ElementType, Shape, Tensor};
use std::fs;
use util::is_version_pre_2024_2;

#[test]
fn memory_safety() -> anyhow::Result<()> {
    // OpenVINO 2024.2 changed the order of the `ov_element_type_e` enum, breaking compatibility
    // with older versions. Since we are using 2024.2+ bindings here, we skip this test when
    // using older libraries.
    if is_version_pre_2024_2() {
        eprintln!("> skipping test due to pre-2024.2 OpenVINO version");
        return Ok(());
    }

    let mut core = Core::new()?;
    let xml = fs::read_to_string(fixture::graph())?;
    let weights = fs::read(fixture::weights())?;

    // Copy the fixture weights into a tensor. Once we're done here we want to get rid of the
    // original weights buffer as a sanity check.
    let shape = Shape::new(&[1, weights.len() as i64])?;
    let mut weights_tensor = Tensor::new(ElementType::U8, &shape)?;
    weights_tensor.get_raw_data_mut()?.copy_from_slice(&weights);
    drop(weights);

    // Now create a model from a reference to the weights tensor. We observed segfault crashes when
    // passing weights by value but not by reference.
    let model = core.read_model_from_buffer(xml.as_bytes(), Some(&weights_tensor))?;
    drop(weights_tensor);

    // Here we double-check that the model is usable. Though it has captured a reference to the
    // `weights_tensor` and that tensor has been dropped, whatever OpenVINO is doing internally must
    // be safe enough. See
    // https://github.com/openvinotoolkit/openvino/blob/d840d86905f013d95cccbafaa0ddff266e250f75/src/inference/src/model_reader.cpp#L178.
    assert_eq!(model.get_inputs_len()?, 1);
    assert!(core.compile_model(&model, DeviceType::CPU).is_ok());
    Ok(())
}
