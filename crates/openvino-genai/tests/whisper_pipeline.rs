//! Integration tests for WhisperPipeline (requires OpenVINO GenAI runtime and model fixtures).

mod fixtures;

use fixtures::whisper_tiny as fixture;
use openvino_genai::WhisperPipeline;

#[test]
fn test_create_pipeline() {
    let model_dir_path = fixture::model_dir();
    let model_dir = model_dir_path.to_string_lossy();

    // Skip if GenAI runtime isn't available in this environment.
    if WhisperPipeline::new(&model_dir, "CPU").is_err() {
        eprintln!("SKIP: WhisperPipeline unavailable (GenAI runtime missing?)");
        return;
    }
}
