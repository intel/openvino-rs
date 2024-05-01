use crate::{try_unsafe, util::Result};
use openvino_sys::{ov_output_const_port_t, ov_port_get_any_name};
use std::ffi::CStr;

/// See [`Node`](https://docs.openvino.ai/2023.3/api/c_cpp_api/group__ov__node__c__api.html).
pub struct Node {
    instance: *mut ov_output_const_port_t,
}

impl Node {
    /// Create a new [`Port`] from [`ov_output_const_port_t`].
    pub(crate) fn new(instance: *mut ov_output_const_port_t) -> Self {
        Self { instance }
    }

    /// Get name of a port.
    pub fn get_name(&self) -> Result<String> {
        let mut c_name = std::ptr::null_mut();
        try_unsafe!(ov_port_get_any_name(
            self.instance,
            std::ptr::addr_of_mut!(c_name)
        ))?;
        let rust_name = unsafe { CStr::from_ptr(c_name) }
            .to_string_lossy()
            .into_owned();
        Ok(rust_name)
    }
}
