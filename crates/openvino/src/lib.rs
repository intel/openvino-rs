//! The [openvino] crate provides high-level, ergonomic, safe Rust bindings to OpenVINO. See the
//! repository [README] for more information, such as build instructions.
//!
//! [openvino]: https://crates.io/crates/openvino
//! [README]: https://github.com/intel/openvino-rs
//!
//! Check the loaded version of OpenVINO:
//! ```
//! assert!(openvino::version().starts_with("2.1"))
//! ```
//!
//! Most interaction with OpenVINO begins with instantiating a [Core]:
//! ```
//! let _ = openvino::Core::new(None).expect("to instantiate the OpenVINO library");
//! ```

mod blob;
mod core;
mod error;
mod network;
mod request;
mod tensor_desc;
mod util;

pub use crate::core::Core;
pub use blob::Blob;
pub use error::{InferenceError, LoadingError, SetupError};
pub use network::{CNNNetwork, ExecutableNetwork};
// Re-publish some OpenVINO enums with a conventional Rust naming (see
// `crates/openvino-sys/build.rs`).
pub use openvino_sys::{
    layout_e as Layout, precision_e as Precision, resize_alg_e as ResizeAlgorithm,
};
pub use request::InferRequest;
pub use tensor_desc::TensorDesc;

/// Emit the version string of the OpenVINO C API backing this implementation.
pub fn version() -> String {
    use std::ffi::CStr;
    openvino_sys::load().expect("to have an OpenVINO shared library available");
    let mut ie_version = unsafe { openvino_sys::ie_c_api_version() };
    let str_version = unsafe { CStr::from_ptr(ie_version.api_version) }
        .to_string_lossy()
        .into_owned();
    unsafe { openvino_sys::ie_version_free(&mut ie_version as *mut openvino_sys::ie_version_t) };
    str_version
}
