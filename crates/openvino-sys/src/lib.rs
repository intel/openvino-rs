#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

/// This string contains the path to the OpenVINO binaries on the system on which this crate was
/// built. __Warning__: do not use this on systems other than the system that built `openvino-sys`.
///
/// Its presence here is necessary because OpenVINO loads target-specific libraries on demand for
/// performing inference. To do so, it relies on a `plugin.xml` file that maps targets (e.g. CPU)
/// to the target-specific implementation library. At runtime, it inspects this file and loads the
/// libraries to satisfy the user's specified targets. By default, the `plugin.xml` file and these
/// libraries will be available at this path.
pub const LIBRARY_PATH: &'static str = env!("LIBRARY_PATH");

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;

    #[test]
    fn check_version() {
        let version = unsafe { CStr::from_ptr(ie_c_api_version().api_version) };
        assert_eq!(
            version.to_string_lossy(),
            "2.1.custom_master_a0581d3d8f004c27d7ce118a91be18b7c8cafb87"
        );
    }
}
