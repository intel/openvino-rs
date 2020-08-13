//! These tests demonstrate how to setup OpenVINO networks.
mod fixture;

use fixture::Fixture;
use openvino::Core;

#[test]
fn read_network() {
    let mut core = Core::new(None).unwrap();
    core.read_network_from_file(
        &Fixture::graph().to_string_lossy(),
        &Fixture::weights().to_string_lossy(),
    )
    .unwrap();
}
