#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

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
