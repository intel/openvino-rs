//! Integration tests for WhisperPipeline (requires OpenVINO GenAI runtime and model fixtures).

mod fixtures;

use fixtures::whisper_tiny as fixture;
use openvino_genai::WhisperPipeline;

#[test]
fn test_create_pipeline() {
    let model_dir = fixture::model_dir();
    let _pipeline = WhisperPipeline::new(&model_dir.to_string_lossy(), "CPU").unwrap();
}
