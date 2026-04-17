//! Integration tests for LlmPipeline (requires OpenVINO GenAI runtime and model fixtures).

mod fixtures;

use fixtures::qwen3 as fixture;
use openvino_genai::{ChatHistory, ChatMessage, LlmPipeline};

fn try_pipeline() -> Option<LlmPipeline> {
    let model_dir = fixture::model_dir();
    let model_path = model_dir.to_string_lossy();

    match LlmPipeline::new(&model_path, "CPU") {
        Ok(p) => Some(p),
        Err(e) => {
            eprintln!("Skipping GenAI LlmPipeline tests: failed to create pipeline: {e}");
            None
        }
    }
}

#[test]
fn test_create_pipeline() {
    let _pipeline = match try_pipeline() {
        Some(p) => p,
        None => return, // skip
    };
}

#[test]
fn test_generate() {
    let mut pipeline = match try_pipeline() {
        Some(p) => p,
        None => return, // skip
    };

    let results = pipeline.generate("What is 2+2?", None, None).unwrap();
    let text = results.get_string().unwrap();
    assert!(!text.is_empty(), "expected non-empty generation output");
}

#[test]
fn test_generate_with_chat_history() {
    let mut pipeline = match try_pipeline() {
        Some(p) => p,
        None => return, // skip
    };

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
