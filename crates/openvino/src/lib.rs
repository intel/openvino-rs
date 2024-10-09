//! The [openvino] crate provides high-level, ergonomic, safe Rust bindings to OpenVINO. See the
//! repository [README] for more information, such as build instructions.
//!
//! [openvino]: https://crates.io/crates/openvino
//! [README]: https://github.com/intel/openvino-rs
//!
//! Check the loaded version of OpenVINO:
//! ```
//! assert!(openvino::version().build_number.starts_with("2"))
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
mod device_type;
mod dimension;
mod element_type;
mod error;
mod layout;
mod model;
mod node;
mod partial_shape;
pub mod prepostprocess;
mod property;
mod rank;
mod request;
mod resize_algorithm;
mod shape;
mod tensor;
mod util;
mod version;

pub use crate::core::Core;
pub use device_type::DeviceType;
pub use dimension::Dimension;
pub use element_type::ElementType;
pub use error::{InferenceError, LoadingError, SetupError};
pub use layout::Layout;
pub use model::{CompiledModel, Model};
pub use node::Node;
pub use partial_shape::PartialShape;
pub use property::{PropertyKey, RwPropertyKey};
pub use rank::Rank;
pub use request::InferRequest;
pub use resize_algorithm::ResizeAlgorithm;
pub use shape::Shape;
pub use tensor::Tensor;
pub use version::{version, Version};
