//! The [openvino] crate provides high-level, ergonomic, safe Rust bindings to OpenVINO. See the
//! repository [README] for more information, such as build instructions.
//!
//! [openvino]: https://crates.io/crates/openvino
//! [README]: https://github.com/intel/openvino-rs
//!
//! Check the loaded version of OpenVINO:
//! ```
//! assert!(openvino::version().starts_with("2"))
//! ```
//!
//! Most interaction with OpenVINO begins with instantiating a [Core]:
//! ```
//! let _ = openvino::Core::new().expect("to instantiate the OpenVINO library");
//! ```

#![deny(missing_docs)]
#![deny(clippy::all)]
#![warn(clippy::pedantic)]
#![warn(clippy::cargo)]
#![allow(
    clippy::must_use_candidate,
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::len_without_is_empty
)]

mod core;
mod element_type;
mod error;
mod layout;
mod model;
mod node;
pub mod prepostprocess;
mod request;
mod shape;
mod tensor;
mod util;

pub use crate::core::Core;
pub use element_type::ElementType;
pub use error::{InferenceError, LoadingError, SetupError};
pub use layout::Layout;
pub use model::{CompiledModel, Model};
pub use node::Node;
pub use request::InferRequest;
pub use shape::Shape;
pub use tensor::Tensor;

/// Emit the version string of the OpenVINO C API backing this implementation.
///
/// # Panics
///
/// Panics if no OpenVINO library can be found.
pub fn version() -> String {
    use std::ffi::CStr;
    openvino_sys::load().expect("to have an OpenVINO shared library available");
    let mut ov_version = openvino_sys::ov_version_t {
        // Initialize the fields to default values
        description: std::ptr::null(),
        buildNumber: std::ptr::null(),
    };
    let code = unsafe { openvino_sys::ov_get_openvino_version(&mut ov_version) };
    assert_eq!(code, 0);
    let version_ptr = { ov_version }.buildNumber;
    let c_str_version = unsafe { CStr::from_ptr(version_ptr) };
    let string_version = c_str_version.to_string_lossy().into_owned();
    unsafe { openvino_sys::ov_version_free(std::ptr::addr_of_mut!(ov_version)) };
    string_version
}
