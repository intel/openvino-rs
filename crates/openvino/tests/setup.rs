//! These tests demonstrate how to setup OpenVINO networks.
mod fixtures;

use fixtures::alexnet::Fixture;
use openvino::Core;

#[test]
fn read_network() {
    let mut core = Core::new().unwrap();
    let read_model = core
        .read_model_from_file(Fixture::graph(), Fixture::weights())
        .unwrap();

    // Check the number of inputs and outputs.
    assert_eq!(read_model.input_size(), Ok(1));
    assert_eq!(read_model.output_size(), Ok(1));
}
