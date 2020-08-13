use std::path::PathBuf;

/// This structure encodes the paths necessary for running the classification example.
pub struct Fixture;
#[allow(dead_code)]
impl Fixture {
    fn dir() -> PathBuf {
        // This seems a bit brittle but works better than `PathBuf::from(file!()).parent()` which is
        // relative to the workspace; when the integration tests are run they are run in the
        // `crates/openvino` directory.
        PathBuf::from("tests/fixture").canonicalize().unwrap()
    }
    pub fn graph() -> PathBuf {
        Fixture::dir().join("frozen_inference_graph.xml")
    }
    pub fn weights() -> PathBuf {
        Fixture::dir().join("frozen_inference_graph.bin")
    }
    pub fn image() -> PathBuf {
        Fixture::dir().join("val2017/000000062808.jpg")
    }
}
