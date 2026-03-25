//! A collection of utility types and macros for use inside this crate.
use crate::InferenceError;

/// This alias makes the implementation slightly less verbose.
pub(crate) type Result<T> = std::result::Result<T, InferenceError>;

/// Convert a Rust string into a string to pass across the C boundary.
#[doc(hidden)]
#[macro_export]
macro_rules! cstr {
    ($str: expr) => {
        std::ffi::CString::new($str).expect("a valid C string")
    };
}

/// Convert an unsafe call to openvino-genai-sys into an [`InferenceError`].
#[doc(hidden)]
#[macro_export]
macro_rules! try_unsafe {
    ($e: expr) => {
        $crate::InferenceError::convert(unsafe { $e })
    };
}

/// Drop one of the Rust wrapper structures using the provided free function. This relies on all
/// Rust wrapper functions having a `ptr` field pointing to their OpenVINO GenAI C structure.
#[doc(hidden)]
#[macro_export]
macro_rules! drop_using_function {
    ($ty: ty, $free_fn: expr) => {
        impl Drop for $ty {
            fn drop(&mut self) {
                let free = $free_fn;
                unsafe { free(self.ptr.cast()) }
            }
        }
    };
}

/// Helper to read a C string using the two-call pattern (first call gets size, second gets data).
///
/// `$get_fn` must be a function with signature: `(source, buffer, &mut size) -> ov_status_e`
pub(crate) unsafe fn get_c_string_two_call<F>(get_fn: F) -> Result<String>
where
    F: Fn(*mut ::std::os::raw::c_char, *mut usize) -> openvino_genai_sys::ov_status_e,
{
    // First call: get the required buffer size.
    let mut size: usize = 0;
    InferenceError::convert(get_fn(std::ptr::null_mut(), &mut size))?;

    if size == 0 {
        return Ok(String::new());
    }

    // Second call: fill the buffer.
    let mut buf = vec![0u8; size];
    InferenceError::convert(get_fn(buf.as_mut_ptr().cast(), &mut size))?;

    // Remove the null terminator if present.
    if buf.last() == Some(&0) {
        buf.pop();
    }
    Ok(String::from_utf8_lossy(&buf).into_owned())
}
