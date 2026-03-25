//! Safe wrapper for the streamer callback mechanism.

use openvino_genai_sys::{ov_genai_streaming_status_e, streamer_callback};
use std::ffi::CStr;
use std::os::raw::{c_char, c_void};

/// The status returned from a streaming callback to control generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamingStatus {
    /// Continue generating tokens.
    Running,
    /// Stop generation, keeping history as-is.
    Stop,
    /// Cancel generation, dropping the last prompt and generated tokens.
    Cancel,
}

impl From<StreamingStatus> for ov_genai_streaming_status_e {
    fn from(status: StreamingStatus) -> Self {
        match status {
            StreamingStatus::Running => ov_genai_streaming_status_e::RUNNING,
            StreamingStatus::Stop => ov_genai_streaming_status_e::STOP,
            StreamingStatus::Cancel => ov_genai_streaming_status_e::CANCEL,
        }
    }
}

/// A streaming callback that receives tokens as they are generated.
///
/// Wraps a Rust closure into a C-compatible [`streamer_callback`].
pub struct Streamer {
    _callback: Box<dyn FnMut(&str) -> StreamingStatus>,
    pub(crate) raw: streamer_callback,
}

/// The extern "C" trampoline that bridges the C callback to the Rust closure.
unsafe extern "C" fn trampoline(str_: *const c_char, args: *mut c_void) -> ov_genai_streaming_status_e {
    let callback = &mut *(args.cast::<Box<dyn FnMut(&str) -> StreamingStatus>>());
    let c_str = CStr::from_ptr(str_);
    let s = c_str.to_string_lossy();
    callback(&s).into()
}

impl Streamer {
    /// Create a new streamer from a closure that receives each generated token string.
    ///
    /// The closure should return a [`StreamingStatus`] to control generation:
    /// - [`StreamingStatus::Running`] to continue
    /// - [`StreamingStatus::Stop`] to stop but keep history
    /// - [`StreamingStatus::Cancel`] to stop and discard the last generation
    pub fn new<F>(callback: F) -> Self
    where
        F: FnMut(&str) -> StreamingStatus + 'static,
    {
        let mut boxed: Box<dyn FnMut(&str) -> StreamingStatus> = Box::new(callback);
        let args = std::ptr::addr_of_mut!(boxed).cast::<c_void>();
        let raw = streamer_callback {
            callback_func: Some(trampoline),
            args,
        };
        Self {
            _callback: boxed,
            raw,
        }
    }

    /// Get a pointer to the raw C callback struct.
    pub(crate) fn as_raw(&self) -> *const streamer_callback {
        &self.raw
    }
}
