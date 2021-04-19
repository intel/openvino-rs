#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

#[macro_use]
mod linking;

mod generated;
pub use generated::*;

use std::path::PathBuf;

/// Return the location of the shared library `openvino-sys` will link to. If compiled with runtime
/// linking, this will attempt to discover the location of a `inference_engine_c_api` shared library
/// on the system. Otherwise (with dynamic linking or compilation from source), this relies on a
/// static path discovered at build time.
///
/// Knowing the location of the OpenVINO libraries is critical to avoid errors, unfortunately.
/// OpenVINO loads target-specific libraries on demand for performing inference. To do so, it relies
/// on a `plugins.xml` file that maps targets (e.g. CPU) to the target-specific implementation
/// library. At runtime, users must pass the path to this file so that OpenVINO can inspect it and
/// load the required libraries to satisfy the user's specified targets. By default, the
/// `plugins.xml` file is found in the same directory as the libraries, e.g.
/// `find().unwrap().parent()`.
pub fn find() -> Option<PathBuf> {
    if cfg!(feature = "runtime-linking") {
        openvino_finder::find("inference_engine_c_api")
    } else {
        Some(PathBuf::from(env!("OPENVINO_LIB_PATH")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;

    #[test]
    fn check_version() {
        load().expect("to have an OpenVINO library available");
        let version = unsafe { CStr::from_ptr(ie_c_api_version().api_version) };
        assert!(version.to_string_lossy().starts_with("2.1"));
    }
}
