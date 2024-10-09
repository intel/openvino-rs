use openvino_sys::{ov_status_e, ov_version_t};

/// Emit the version of the OpenVINO C library backing this implementation.
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
    assert_eq!(code, ov_status_e::OK);
    let version = Version::from(&ov_version);
    unsafe { openvino_sys::ov_version_free(std::ptr::addr_of_mut!(ov_version)) };
    version
}

/// See [`ov_version`](https://docs.openvino.ai/2024/api/c_cpp_api/structov__version.html).
pub struct Version {
    /// A string representing OpenVINO version.
    pub build_number: String,
    /// A string representing OpenVINO description.
    pub description: String,
}

impl From<&ov_version_t> for Version {
    fn from(ov_version: &ov_version_t) -> Self {
        let c_str_version = unsafe { std::ffi::CStr::from_ptr(ov_version.buildNumber) };
        let c_str_description = unsafe { std::ffi::CStr::from_ptr(ov_version.description) };
        Self {
            build_number: c_str_version.to_string_lossy().into_owned(),
            description: c_str_description.to_string_lossy().into_owned(),
        }
    }
}

impl Version {
    /// Parse the version into its parts.
    pub fn parts(&self) -> impl Iterator<Item = &str> {
        self.build_number.split(['.', '-'])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_parts() -> anyhow::Result<()> {
        let version = version();
        let year: usize = version.parts().next().unwrap().parse()?;
        assert!(year > 2020);
        Ok(())
    }
}
