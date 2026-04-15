//! Safe wrapper for [`ov_genai_json_container`].

use crate::{cstr, util::Result, InferenceError};
use openvino_genai_sys::{
    self, ov_genai_json_container, ov_genai_json_container_create_from_json_string,
    ov_genai_json_container_free, ov_genai_json_container_status_e,
    ov_genai_json_container_to_json_string,
};

/// A JSON container for structured data exchange with the GenAI C API.
///
/// Used primarily for constructing chat messages and tool definitions.
pub struct JsonContainer {
    ptr: *mut ov_genai_json_container,
}

impl Drop for JsonContainer {
    fn drop(&mut self) {
        unsafe { ov_genai_json_container_free(self.ptr) }
    }
}

/// Convert a JSON container status code to a Result.
fn convert_json_status(status: ov_genai_json_container_status_e) -> Result<()> {
    match status {
        ov_genai_json_container_status_e::OV_GENAI_JSON_CONTAINER_OK => Ok(()),
        ov_genai_json_container_status_e::OV_GENAI_JSON_CONTAINER_INVALID_PARAM => {
            Err(InferenceError::InvalidCParam)
        }
        ov_genai_json_container_status_e::OV_GENAI_JSON_CONTAINER_INVALID_JSON => {
            Err(InferenceError::ParameterMismatch)
        }
        ov_genai_json_container_status_e::OV_GENAI_JSON_CONTAINER_OUT_OF_BOUNDS => {
            Err(InferenceError::OutOfBounds)
        }
        ov_genai_json_container_status_e::OV_GENAI_JSON_CONTAINER_ERROR => {
            Err(InferenceError::GeneralError)
        }
    }
}

impl JsonContainer {
    /// Create a JSON container from a JSON string.
    ///
    /// # Example
    ///
    /// ```
    /// openvino_genai::load().unwrap();
    /// let msg = openvino_genai::JsonContainer::from_json_str(
    ///     r#"{"role": "user", "content": "Hello"}"#,
    /// ).unwrap();
    /// ```
    pub fn from_json_str(json: &str) -> Result<Self> {
        let json = cstr!(json);
        let mut ptr = std::ptr::null_mut();
        convert_json_status(unsafe {
            ov_genai_json_container_create_from_json_string(
                std::ptr::addr_of_mut!(ptr),
                json.as_ptr(),
            )
        })?;
        Ok(Self { ptr })
    }

    /// Convert this container to a JSON string.
    pub fn to_json_string(&self) -> Result<String> {
        // Two-call pattern: first get size, then fill buffer.
        let mut size: usize = 0;
        convert_json_status(unsafe {
            ov_genai_json_container_to_json_string(self.ptr, std::ptr::null_mut(), &mut size)
        })?;

        if size == 0 {
            return Ok(String::new());
        }

        let mut buf = vec![0u8; size];
        convert_json_status(unsafe {
            ov_genai_json_container_to_json_string(self.ptr, buf.as_mut_ptr().cast(), &mut size)
        })?;

        if buf.last() == Some(&0) {
            buf.pop();
        }
        Ok(String::from_utf8_lossy(&buf).into_owned())
    }

    /// Get the raw pointer. For internal use.
    pub(crate) fn as_ptr(&self) -> *const ov_genai_json_container {
        self.ptr
    }

    /// Construct from a raw pointer. For internal use.
    pub(crate) fn from_raw_ptr(ptr: *mut ov_genai_json_container) -> Self {
        Self { ptr }
    }
}
