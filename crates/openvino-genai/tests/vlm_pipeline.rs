//! Integration tests for VlmPipeline (requires OpenVINO GenAI runtime and model fixtures).

mod fixtures;

use fixtures::internvl2 as fixture;
use openvino_genai::VlmPipeline;

#[test]
fn test_create_pipeline() {
    let model_dir = fixture::model_dir();
    let _pipeline = VlmPipeline::new(&model_dir.to_string_lossy(), "CPU").unwrap();
}
