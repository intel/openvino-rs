//! Integration tests for LlmPipeline (requires OpenVINO GenAI runtime and model fixtures).

mod fixtures;

use fixtures::qwen3 as fixture;
use openvino_genai::{ChatHistory, ChatMessage, LlmPipeline};

#[test]
fn test_create_pipeline() {
    let model_dir = fixture::model_dir();
    let _pipeline = LlmPipeline::new(&model_dir.to_string_lossy(), "CPU").unwrap();
}

#[test]
fn test_generate() {
    let model_dir = fixture::model_dir();
    let mut pipeline = LlmPipeline::new(&model_dir.to_string_lossy(), "CPU").unwrap();
    let results = pipeline.generate("What is 2+2?", None, None).unwrap();
    let text = results.get_string().unwrap();
    assert!(!text.is_empty(), "expected non-empty generation output");
}

#[test]
fn test_generate_with_chat_history() {
    let model_dir = fixture::model_dir();
    let mut pipeline = LlmPipeline::new(&model_dir.to_string_lossy(), "CPU").unwrap();

    let mut history = ChatHistory::new().unwrap();
    history
        .push(&ChatMessage::system("You are a helpful assistant."))
        .unwrap();
    history.push(&ChatMessage::user("What is 2+2?")).unwrap();

    let results = pipeline
        .generate_with_history(&history, None, None)
        .unwrap();
    let text = results.get_string().unwrap();
    assert!(!text.is_empty(), "expected non-empty generation output");
}
