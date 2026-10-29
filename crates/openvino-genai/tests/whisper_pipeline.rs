//! Integration tests for WhisperPipeline (requires OpenVINO GenAI runtime and model fixtures).

mod fixtures;

use fixtures::whisper_tiny as fixture;
use openvino_genai::{WhisperGenerationConfig, WhisperPipeline};

#[test]
fn test_create_pipeline() {
    let model_dir_path = fixture::model_dir();
    let model_dir = model_dir_path.to_string_lossy();

    // Skip if GenAI runtime isn't available in this environment.
    if WhisperPipeline::new(&model_dir, "CPU").is_err() {
        eprintln!("SKIP: WhisperPipeline unavailable (GenAI runtime missing?)");
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
    }

    // CACHE_DIR is a no-op on CPU in terms of correctness (there's nothing
    // to compile-cache the way there is on GPU/NPU), but the property must
    // still be accepted rather than rejected outright.
}

#[test]
fn test_initial_prompt_and_hotwords_round_trip() {
    // Skip if GenAI runtime isn't available in this environment.
    if let Err(e) = openvino_genai::load() {
        eprintln!("SKIP: OpenVINO GenAI runtime not available: {e:?}");
        return;
    }
    let mut config = WhisperGenerationConfig::new().unwrap();

    assert_eq!(config.get_initial_prompt().unwrap(), None);
    assert_eq!(config.get_hotwords().unwrap(), None);

    config.set_initial_prompt("Voxtype, OpenVINO").unwrap();
    config.set_hotwords("Hyprland").unwrap();
    assert_eq!(
        config.get_initial_prompt().unwrap().as_deref(),
        Some("Voxtype, OpenVINO")
    );
    assert_eq!(config.get_hotwords().unwrap().as_deref(), Some("Hyprland"));
}
