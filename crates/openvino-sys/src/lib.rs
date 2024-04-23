//! This crate provides low-level, unsafe, Rust bindings to OpenVINO™ using its [C API]. If you are
//! looking to use OpenVINO™ from Rust, you likely should look at the ergonomic, safe bindings in
//! [openvino], which depends on this crate. See the repository [README] for more information,
//! including build instructions.
//!
//! [C API]: https://docs.openvinotoolkit.org/2020.1/ie_c_api/groups.html
//! [openvino-sys]: https://crates.io/crates/openvino-sys
//! [openvino]: https://crates.io/crates/openvino
//! [README]: https://github.com/intel/openvino-rs/tree/main/crates/openvino-sys
//!
//! An example interaction with raw [openvino-sys]:
//! ```
//! # use std::ffi::CStr;
//! openvino_sys::library::load().expect("to have an OpenVINO library available");
//! let mut ov_version = openvino_sys::ov_version_t {
//!     // Initialize the fields to default values
//!     description: std::ptr::null(),
//!     buildNumber: std::ptr::null(),
//! };
//! let code = unsafe { openvino_sys::ov_get_openvino_version(&mut ov_version) };
//! assert_eq!(code, 0);
//! let version_ptr = { ov_version }.buildNumber;
//! let string_version = unsafe { CStr::from_ptr(version_ptr) }.to_string_lossy().into_owned();
//! unsafe { openvino_sys::ov_version_free(std::ptr::addr_of_mut!(ov_version)) };
//! assert!(string_version.starts_with("2"));
//! ```

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]
#![allow(unused, dead_code)]
#![deny(clippy::all)]
#![warn(clippy::pedantic)]
#![warn(clippy::cargo)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::wildcard_imports)]

mod linking;

mod generated;
pub use generated::*;

/// Contains extra utilities for finding and loading the OpenVINO shared libraries.
pub mod library {
    use std::path::PathBuf;

    // Include the definition of `load` here. This allows hiding all of the "extra" linking-related
    // functions in the same place, without polluting the top-level namespace (which should only
    // contain foreign functions and types).
    #[doc(inline)]
    pub use super::generated::load;

    /// Return the location of the shared library `openvino-sys` will link to. If compiled with
    /// runtime linking, this will attempt to discover the location of a `openvino_c` shared library
    /// on the system. Otherwise (with dynamic linking or compilation from source), this relies on a
    /// static path discovered at build time.
    ///
    /// Knowing the location of the OpenVINO libraries is critical to avoid errors, unfortunately.
    /// OpenVINO loads target-specific libraries on demand for performing inference. To do so, it
    /// relies on a `plugins.xml` file that maps targets (e.g. CPU) to the target-specific
    /// implementation library. At runtime, users must pass the path to this file so that OpenVINO
    /// can inspect it and load the required libraries to satisfy the user's specified targets. By
    /// default, the `plugins.xml` file is found in the same directory as the libraries, e.g.
    /// `find().unwrap().parent()`.
    pub fn find() -> Option<PathBuf> {
        if cfg!(feature = "runtime-linking") {
            openvino_finder::find("openvino_c")
        } else {
            Some(PathBuf::from(env!("OPENVINO_LIB_PATH")))
        }
    }
}
