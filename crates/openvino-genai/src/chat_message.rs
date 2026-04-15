//! Typed chat message types for use with [`ChatHistory`](crate::ChatHistory).

use std::fmt::Write as _;

use crate::{util::Result, JsonContainer};

/// A single tool call requested by the model.
#[derive(Debug, Clone)]
pub struct ToolCall {
    /// The name of the function to call.
    pub name: String,
    /// The arguments as a JSON string (e.g., `{"location": "New York"}`).
    pub arguments: String,
}

/// A chat message with typed variants for each role.
///
/// # Examples
///
/// ```
/// openvino_genai::load().unwrap();
/// use openvino_genai::{ChatMessage, ChatHistory};
///
/// let mut history = ChatHistory::new().unwrap();
/// history.push(&ChatMessage::system("You are a helpful assistant.")).unwrap();
/// history.push(&ChatMessage::user("What is the weather in Paris?")).unwrap();
/// history.push(&ChatMessage::assistant("The weather in Paris is sunny.")).unwrap();
/// ```
#[derive(Debug, Clone)]
pub enum ChatMessage {
    /// A system prompt or instruction.
    System {
        /// The system message content.
        content: String,
    },
    /// A user message.
    User {
        /// The user message content.
        content: String,
    },
    /// An assistant response, optionally containing tool calls.
    Assistant {
        /// The assistant message content.
        content: String,
        /// Tool calls requested by the model (empty if none).
        tool_calls: Vec<ToolCall>,
    },
    /// A tool/function result returned to the model.
    Tool {
        /// The tool result content.
        content: String,
        /// The ID of the tool call this result corresponds to.
        tool_call_id: String,
    },
}

impl ChatMessage {
    /// Create a system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self::System {
            content: content.into(),
        }
    }

    /// Create a user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self::User {
            content: content.into(),
        }
    }

    /// Create an assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::Assistant {
            content: content.into(),
            tool_calls: Vec::new(),
        }
    }

    /// Create an assistant message that includes tool calls.
    pub fn assistant_with_tool_calls(
        content: impl Into<String>,
        tool_calls: Vec<ToolCall>,
    ) -> Self {
        Self::Assistant {
            content: content.into(),
            tool_calls,
        }
    }

    /// Create a tool result message.
    pub fn tool(content: impl Into<String>, tool_call_id: impl Into<String>) -> Self {
        Self::Tool {
            content: content.into(),
            tool_call_id: tool_call_id.into(),
        }
    }

    /// Serialize this message to a JSON string matching the format expected by the C API.
    pub(crate) fn to_json_string(&self) -> String {
        match self {
            Self::System { content } => {
                format!(r#"{{"role":"system","content":{}}}"#, json_escape(content))
            }
            Self::User { content } => {
                format!(r#"{{"role":"user","content":{}}}"#, json_escape(content))
            }
            Self::Assistant {
                content,
                tool_calls,
            } => {
                if tool_calls.is_empty() {
                    format!(
                        r#"{{"role":"assistant","content":{}}}"#,
                        json_escape(content)
                    )
                } else {
                    let calls: Vec<String> = tool_calls
                        .iter()
                        .map(|tc| {
                            format!(
                                r#"{{"name":{},"arguments":{}}}"#,
                                json_escape(&tc.name),
                                // arguments is already a JSON string, embed it directly
                                &tc.arguments
                            )
                        })
                        .collect();
                    format!(
                        r#"{{"role":"assistant","content":{},"tool_calls":[{}]}}"#,
                        json_escape(content),
                        calls.join(",")
                    )
                }
            }
            Self::Tool {
                content,
                tool_call_id,
            } => {
                format!(
                    r#"{{"role":"tool","content":{},"tool_call_id":{}}}"#,
                    json_escape(content),
                    json_escape(tool_call_id)
                )
            }
        }
    }

    /// Convert this message to a [`JsonContainer`] for the C API.
    pub(crate) fn to_json_container(&self) -> Result<JsonContainer> {
        JsonContainer::from_json_str(&self.to_json_string())
    }
}

/// Escape a string for embedding in JSON. Returns a quoted JSON string value.
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str(r#"\""#),
            '\\' => out.push_str(r"\\"),
            '\n' => out.push_str(r"\n"),
            '\r' => out.push_str(r"\r"),
            '\t' => out.push_str(r"\t"),
            c if c < '\x20' => {
                // Control characters as \u00XX
                for byte in c.encode_utf8(&mut [0; 4]).bytes() {
                    let _ = write!(out, r"\u{byte:04x}");
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_escape_basic() {
        assert_eq!(json_escape("hello"), r#""hello""#);
        assert_eq!(json_escape(r#"say "hi""#), r#""say \"hi\"""#);
        assert_eq!(json_escape("line\nnewline"), r#""line\nnewline""#);
    }

    #[test]
    fn json_escape_backslash() {
        assert_eq!(json_escape(r"a\b"), r#""a\\b""#);
    }

    #[test]
    fn system_message_json() {
        let msg = ChatMessage::system("You are helpful");
        assert_eq!(
            msg.to_json_string(),
            r#"{"role":"system","content":"You are helpful"}"#
        );
    }

    #[test]
    fn user_message_json() {
        let msg = ChatMessage::user("Hello");
        assert_eq!(msg.to_json_string(), r#"{"role":"user","content":"Hello"}"#);
    }

    #[test]
    fn assistant_message_json() {
        let msg = ChatMessage::assistant("Hi there");
        assert_eq!(
            msg.to_json_string(),
            r#"{"role":"assistant","content":"Hi there"}"#
        );
    }

    #[test]
    fn assistant_tool_calls_json() {
        let msg = ChatMessage::assistant_with_tool_calls(
            "",
            vec![ToolCall {
                name: "get_weather".into(),
                arguments: r#"{"location":"New York"}"#.into(),
            }],
        );
        assert_eq!(
            msg.to_json_string(),
            r#"{"role":"assistant","content":"","tool_calls":[{"name":"get_weather","arguments":{"location":"New York"}}]}"#
        );
    }

    #[test]
    fn tool_message_json() {
        let msg = ChatMessage::tool("72°F and sunny", "call_1");
        assert_eq!(
            msg.to_json_string(),
            r#"{"role":"tool","content":"72°F and sunny","tool_call_id":"call_1"}"#
        );
    }

    #[test]
    fn message_with_special_chars() {
        let msg = ChatMessage::user("She said \"hello\"\nand then\\left");
        assert_eq!(
            msg.to_json_string(),
            r#"{"role":"user","content":"She said \"hello\"\nand then\\left"}"#
        );
    }
}
