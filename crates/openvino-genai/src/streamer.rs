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
            StreamingStatus::Running => ov_genai_streaming_status_e::OV_GENAI_STREAMING_STATUS_RUNNING,
            StreamingStatus::Stop => ov_genai_streaming_status_e::OV_GENAI_STREAMING_STATUS_STOP,
            StreamingStatus::Cancel => ov_genai_streaming_status_e::OV_GENAI_STREAMING_STATUS_CANCEL,
        }
    }
}

/// A streaming callback that receives tokens as they are generated.
///
/// Wraps a Rust closure into a C-compatible [`streamer_callback`].
///
/// The callback is heap-allocated via a double-`Box` so that the pointer passed
/// to C (`args`) remains stable regardless of where the `Streamer` struct itself
/// moves. The outer `Box` is leaked into a raw pointer that the trampoline
/// dereferences; it is freed in `Drop`.
pub struct Streamer {
    /// Stable heap pointer to the boxed closure. Passed as `args` to C.
    callback_ptr: *mut Box<dyn FnMut(&str) -> StreamingStatus>,
}

/// The extern "C" trampoline that bridges the C callback to the Rust closure.
unsafe extern "C" fn trampoline(str_: *const c_char, args: *mut c_void) -> ov_genai_streaming_status_e {
    let callback = &mut *(args.cast::<Box<dyn FnMut(&str) -> StreamingStatus>>());
    let c_str = CStr::from_ptr(str_);
    let s = c_str.to_string_lossy();
    callback(&s).into()
}

impl Drop for Streamer {
    fn drop(&mut self) {
        // Reclaim the leaked outer Box.
        unsafe { drop(Box::from_raw(self.callback_ptr)); }
    }
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
        let boxed: Box<dyn FnMut(&str) -> StreamingStatus> = Box::new(callback);
        // Double-box: the outer Box gives us a stable heap pointer that survives moves.
        let callback_ptr = Box::into_raw(Box::new(boxed));
        Self { callback_ptr }
    }

    /// Build a C-compatible [`streamer_callback`] struct.
    ///
    /// The returned struct borrows from `self` and must not outlive it.
    pub(crate) fn as_raw(&self) -> streamer_callback {
        streamer_callback {
            callback_func: Some(trampoline),
            args: self.callback_ptr.cast::<c_void>(),
        }
    }
}
