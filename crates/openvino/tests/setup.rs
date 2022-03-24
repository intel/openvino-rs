//! These tests demonstrate how to setup OpenVINO networks.
mod fixtures;

use fixtures::alexnet::Fixture;
use openvino::Core;
use std::fs;

#[test]
fn read_network() {
    let mut core = Core::new(None).unwrap();
    let model = fs::read(Fixture::graph()).unwrap();
    let weights = fs::read(Fixture::weights()).unwrap();
    let network = core.read_network_from_buffer(&model, &weights).unwrap();

    // Check the number of inputs and outputs.
    assert_eq!(network.get_inputs_len(), Ok(1));
    assert_eq!(network.get_outputs_len(), Ok(1));
}
