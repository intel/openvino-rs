use openvino_sys::ov_version_t;

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
