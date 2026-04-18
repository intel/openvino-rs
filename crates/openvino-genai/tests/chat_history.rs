//! Integration tests for ChatHistory (requires OpenVINO GenAI runtime).

use openvino_genai::{ChatHistory, ChatMessage, JsonContainer};

pub fn genai_available() -> bool {
    match openvino_genai::load() {
        Ok(()) => true,
        Err(e) => {
            eprintln!("Skipping test: OpenVINO GenAI runtime not available: {e:?}");
            false
        }
    }
}

#[test]
fn test_create_and_size() {
    if !genai_available() {
        return;
    }
    let history = ChatHistory::new().unwrap();
    assert_eq!(history.size().unwrap(), 0);
}

#[test]
fn test_push_typed_and_size() {
    if !genai_available() {
        return;
    }
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
    if !genai_available() {
        return;
    }
    let mut history = ChatHistory::new().unwrap();

    let msg = JsonContainer::from_json_str(r#"{"role": "user", "content": "Hello"}"#).unwrap();
    history.push_back(&msg).unwrap();
    assert_eq!(history.size().unwrap(), 1);
}

#[test]
fn test_clear() {
    if !genai_available() {
        return;
    }
    let mut history = ChatHistory::new().unwrap();

    history.push(&ChatMessage::user("Hello")).unwrap();
    history.push(&ChatMessage::assistant("Hi")).unwrap();
    assert_eq!(history.size().unwrap(), 2);

    history.clear().unwrap();
    assert_eq!(history.size().unwrap(), 0);
}
