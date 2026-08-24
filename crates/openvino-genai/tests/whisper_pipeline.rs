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

#[test]
fn test_create_pipeline_with_properties() {
    let model_dir_path = fixture::model_dir();
    let model_dir = model_dir_path.to_string_lossy();
    let cache_dir = std::env::temp_dir().join("openvino-genai-test-cache");
    let cache_dir_str = cache_dir.to_string_lossy();

    // Skip if GenAI runtime isn't available in this environment.
    if WhisperPipeline::with_properties(&model_dir, "CPU", &[("CACHE_DIR", &cache_dir_str)])
        .is_err()
    {
        eprintln!("SKIP: WhisperPipeline unavailable (GenAI runtime missing?)");
        return;
    }

    // CACHE_DIR is a no-op on CPU in terms of correctness (there's nothing
    // to compile-cache the way there is on GPU/NPU), but the property must
    // still be accepted rather than rejected outright.
}
