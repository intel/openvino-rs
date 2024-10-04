//! These tests demonstrate how to setup OpenVINO networks.

mod fixtures;
mod util;

use fixtures::alexnet as fixture;
use openvino::{Core, ElementType, Shape, Tensor};
use std::fs;
use util::is_version_pre_2024_2;

#[test]
fn read_network() {
    let mut core = Core::new().unwrap();
    let read_model = core
        .read_model_from_file(
            &fixture::graph().to_string_lossy(),
            &fixture::weights().to_string_lossy(),
        )
        .unwrap();

    // Check the number of inputs and outputs.
    assert_eq!(read_model.get_inputs_len(), Ok(1));
    assert_eq!(read_model.get_outputs_len(), Ok(1));
}

#[test]
fn read_network_from_buffers() {
    // OpenVINO 2024.2 changed the order of the `ov_element_type_e` enum, breaking compatibility
    // with older versions. Since we are using 2024.2+ bindings here, we skip this test when
    // using older libraries.
    if is_version_pre_2024_2() {
        return;
    }

    let mut core = Core::new().unwrap();
    let graph = fs::read(&fixture::graph()).unwrap();
    let weights = {
        let weights = fs::read(&fixture::weights()).unwrap();
        let shape = Shape::new(&[1, weights.len() as i64]).unwrap();
        let mut tensor = Tensor::new(ElementType::U8, &shape).unwrap();
        let buffer = tensor.get_raw_data_mut().unwrap();
        buffer.copy_from_slice(&weights);
        tensor
    };

    let read_model = core.read_model_from_buffer(&graph, Some(&weights)).unwrap();

    // Check the number of inputs and outputs.
    assert_eq!(read_model.get_inputs_len(), Ok(1));
    assert_eq!(read_model.get_outputs_len(), Ok(1));
}
