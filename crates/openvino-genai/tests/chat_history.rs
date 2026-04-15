//! Integration tests for ChatHistory (requires OpenVINO GenAI runtime).

use openvino_genai::{ChatHistory, ChatMessage, JsonContainer};

#[test]
fn test_create_and_size() {
    openvino_genai::load().unwrap();
    let history = ChatHistory::new().unwrap();
    assert_eq!(history.size().unwrap(), 0);
}

#[test]
fn test_push_typed_and_size() {
    openvino_genai::load().unwrap();
    let mut history = ChatHistory::new().unwrap();

    history
        .push(&ChatMessage::system("You are helpful"))
        .unwrap();
    assert_eq!(history.size().unwrap(), 1);

    history.push(&ChatMessage::user("Hello")).unwrap();
    assert_eq!(history.size().unwrap(), 2);

    history.push(&ChatMessage::assistant("Hi there")).unwrap();
    assert_eq!(history.size().unwrap(), 3);
}

#[test]
fn test_push_raw_and_size() {
    openvino_genai::load().unwrap();
    let mut history = ChatHistory::new().unwrap();

    let msg = JsonContainer::from_json_str(r#"{"role": "user", "content": "Hello"}"#).unwrap();
    history.push_back(&msg).unwrap();
    assert_eq!(history.size().unwrap(), 1);
}

#[test]
fn test_clear() {
    openvino_genai::load().unwrap();
    let mut history = ChatHistory::new().unwrap();

    history.push(&ChatMessage::user("Hello")).unwrap();
    history.push(&ChatMessage::assistant("Hi")).unwrap();
    assert_eq!(history.size().unwrap(), 2);

    history.clear().unwrap();
    assert_eq!(history.size().unwrap(), 0);
}
