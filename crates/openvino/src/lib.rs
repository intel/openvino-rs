//! The [openvino] crate provides high-level, ergonomic, safe Rust bindings to OpenVINO. See the
//! repository [README] for more information, such as build instructions.
//!
//! [openvino]: https://crates.io/crates/openvino
//! [README]: https://github.com/intel/openvino-rs

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
mod device_type;
mod element_type;
mod error;
mod layout;
mod model;
mod port;
mod prepostprocess;
mod property;
mod request;
mod resize_algorithm;
mod shape;
mod tensor;
mod util;
mod version;

pub use core::Core;
pub use device_type::DeviceType;
pub use element_type::ElementType;
pub use error::{InferenceError, LoadingError, SetupError};
pub use layout::Layout;
pub use model::{CompiledModel, Model};
pub use port::Port;
pub use prepostprocess::{
    PrePostProcess, PreprocessInputInfo, PreprocessInputModelInfo, PreprocessInputTensorInfo,
    PreprocessOutputInfo, PreprocessOutputTensorInfo, PreprocessSteps,
};
pub use property::{PropertyKey, RwPropertyKey};
pub use request::InferRequest;
pub use resize_algorithm::ResizeAlgorithm;
pub use shape::Shape;
pub use tensor::Tensor;
pub use version::Version;

/// Emit the version string of the OpenVINO C API backing this implementation.
///
/// # Panics
///
/// Panics if no OpenVINO library can be found.
pub fn version() -> Version {
    openvino_sys::load().expect("to have an OpenVINO shared library available");
    let mut ov_version = openvino_sys::ov_version_t {
        buildNumber: std::ptr::null(),
        description: std::ptr::null(),
    };
    let code = unsafe { openvino_sys::ov_get_openvino_version(&mut ov_version) };
    assert_eq!(code, 0);
    let version = Version::from(&ov_version);
    unsafe { openvino_sys::ov_version_free(std::ptr::addr_of_mut!(ov_version)) };
    version
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_core() {
        let _ = Core::new().expect("to instantiate the OpenVINO library");
    }

    #[test]
    fn test_version() {
        assert!(version().build_number.starts_with("2"));
    }
}
