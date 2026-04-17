//! Safe wrapper for [`ov_genai_chat_history`].

use crate::{drop_using_function, util::Result, ChatMessage, InferenceError, JsonContainer};
use openvino_genai_sys::{
    self, ov_genai_chat_history, ov_genai_chat_history_clear, ov_genai_chat_history_create,
    ov_genai_chat_history_free, ov_genai_chat_history_get_extra_context,
    ov_genai_chat_history_get_tools, ov_genai_chat_history_push_back,
    ov_genai_chat_history_set_extra_context, ov_genai_chat_history_set_tools,
    ov_genai_chat_history_size, ov_genai_chat_history_status_e,
};

/// A chat history for multi-turn conversations.
pub struct ChatHistory {
    ptr: *mut ov_genai_chat_history,
}
drop_using_function!(ChatHistory, ov_genai_chat_history_free);

// SAFETY: The underlying C object is not accessed concurrently by the C library;
// callers must ensure exclusive access (e.g., via Mutex) when sharing across threads.
unsafe impl Send for ChatHistory {}

/// Convert a chat history status code to a Result.
fn convert_chat_status(status: ov_genai_chat_history_status_e) -> Result<()> {
    match status {
        ov_genai_chat_history_status_e::OV_GENAI_CHAT_HISTORY_OK => Ok(()),
        ov_genai_chat_history_status_e::OV_GENAI_CHAT_HISTORY_INVALID_PARAM => {
            Err(InferenceError::InvalidCParam)
        }
        ov_genai_chat_history_status_e::OV_GENAI_CHAT_HISTORY_OUT_OF_BOUNDS => {
            Err(InferenceError::OutOfBounds)
        }
        ov_genai_chat_history_status_e::OV_GENAI_CHAT_HISTORY_EMPTY => {
            Err(InferenceError::NotFound)
        }
        ov_genai_chat_history_status_e::OV_GENAI_CHAT_HISTORY_INVALID_JSON => {
            Err(InferenceError::ParameterMismatch)
        }
        ov_genai_chat_history_status_e::OV_GENAI_CHAT_HISTORY_ERROR => {
            Err(InferenceError::GeneralError)
        }
    }
}

impl ChatHistory {
    /// Create a new empty chat history.
    pub fn new() -> Result<Self> {
        let mut ptr = std::ptr::null_mut();
        convert_chat_status(unsafe { ov_genai_chat_history_create(std::ptr::addr_of_mut!(ptr)) })?;
        Ok(Self { ptr })
    }

    /// Add a typed [`ChatMessage`] to the chat history.
    ///
    /// # Example
    ///
    /// ```no_run
    /// openvino_genai::load().unwrap();
    /// use openvino_genai::{ChatMessage, ChatHistory};
    ///
    /// let mut history = ChatHistory::new().unwrap();
    /// history.push(&ChatMessage::user("Hello")).unwrap();
    /// ```
    pub fn push(&mut self, message: &ChatMessage) -> Result<()> {
        let container = message.to_json_container()?;
        self.push_back(&container)
    }

    /// Add a raw [`JsonContainer`] message to the chat history.
    ///
    /// Prefer [`push`](Self::push) for most use cases. This method is available as an escape
    /// hatch for message shapes not covered by [`ChatMessage`].
    pub fn push_back(&mut self, message: &JsonContainer) -> Result<()> {
        convert_chat_status(unsafe { ov_genai_chat_history_push_back(self.ptr, message.as_ptr()) })
    }

    /// Get the number of messages in the history.
    pub fn size(&self) -> Result<usize> {
        let mut size: usize = 0;
        convert_chat_status(unsafe { ov_genai_chat_history_size(self.ptr, &mut size) })?;
        Ok(size)
    }

    /// Clear all messages from the history.
    pub fn clear(&mut self) -> Result<()> {
        convert_chat_status(unsafe { ov_genai_chat_history_clear(self.ptr) })
    }

    /// Set tool definitions for function calling.
    ///
    /// The `tools` parameter should be a [`JsonContainer`] containing an array of tool
    /// definitions (e.g., OpenAI-compatible function schemas). These are incorporated into
    /// the model's chat template during generation.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use openvino_genai::{ChatHistory, JsonContainer};
    ///
    /// let mut history = ChatHistory::new().unwrap();
    /// let tools = JsonContainer::from_json_str(r#"[{
    ///     "type": "function",
    ///     "function": {
    ///         "name": "search_hotkeys",
    ///         "description": "Search for keyboard shortcuts",
    ///         "parameters": {
    ///             "type": "object",
    ///             "properties": { "query": { "type": "string" } },
    ///             "required": ["query"]
    ///         }
    ///     }
    /// }]"#).unwrap();
    /// history.set_tools(&tools).unwrap();
    /// ```
    pub fn set_tools(&mut self, tools: &JsonContainer) -> Result<()> {
        convert_chat_status(unsafe { ov_genai_chat_history_set_tools(self.ptr, tools.as_ptr()) })
    }

    /// Get the current tool definitions.
    ///
    /// Returns a [`JsonContainer`] with the array of tool definitions previously set
    /// via [`set_tools`](Self::set_tools).
    pub fn get_tools(&self) -> Result<JsonContainer> {
        let mut ptr = std::ptr::null_mut();
        convert_chat_status(unsafe {
            ov_genai_chat_history_get_tools(self.ptr, std::ptr::addr_of_mut!(ptr))
        })?;
        Ok(JsonContainer::from_raw_ptr(ptr))
    }

    /// Set extra context for custom template variables.
    ///
    /// The `extra_context` parameter should be a [`JsonContainer`] containing an object
    /// with additional variables to be passed to the model's chat template.
    pub fn set_extra_context(&mut self, extra_context: &JsonContainer) -> Result<()> {
        convert_chat_status(unsafe {
            ov_genai_chat_history_set_extra_context(self.ptr, extra_context.as_ptr())
        })
    }

    /// Get the current extra context.
    ///
    /// Returns a [`JsonContainer`] with the extra context object previously set
    /// via [`set_extra_context`](Self::set_extra_context).
    pub fn get_extra_context(&self) -> Result<JsonContainer> {
        let mut ptr = std::ptr::null_mut();
        convert_chat_status(unsafe {
            ov_genai_chat_history_get_extra_context(self.ptr, std::ptr::addr_of_mut!(ptr))
        })?;
        Ok(JsonContainer::from_raw_ptr(ptr))
    }

    /// Get the raw pointer. For internal use.
    pub(crate) fn as_ptr(&self) -> *const ov_genai_chat_history {
        self.ptr
    }
}
