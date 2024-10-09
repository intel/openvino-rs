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

/// Convert an unsafe call to openvino-sys into an [`InferenceError`].
#[doc(hidden)]
#[macro_export]
macro_rules! try_unsafe {
    ($e: expr) => {
        $crate::InferenceError::convert(unsafe { $e })
    };
}

/// Drop one of the Rust wrapper structures using the provided free function. This relies on all
/// Rust wrapper functions having a `ptr` field pointing to their OpenVINO C structure.
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
