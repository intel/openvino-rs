use crate::{cstr, drop_using_function, try_unsafe, util::Result};
use openvino_sys::{ov_layout_create, ov_layout_free, ov_layout_t};

/// See [`ov_layout_t`](https://docs.openvino.ai/2024/api/c_cpp_api/group__ov__layout__c__api.html).
pub struct Layout {
    ptr: *mut ov_layout_t,
}
drop_using_function!(Layout, ov_layout_free);

impl Layout {
    /// Get a pointer to the [`ov_layout_t`].
    #[inline]
    pub(crate) fn as_mut_ptr(&mut self) -> *mut ov_layout_t {
        self.ptr
    }

    /// Creates a new layout with the given description.
    pub fn new(layout_desc: &str) -> Result<Self> {
        let layout_desc = cstr!(layout_desc);
        let mut layout = std::ptr::null_mut();
        try_unsafe!(ov_layout_create(
            layout_desc.as_ptr(),
            std::ptr::addr_of_mut!(layout)
        ))?;
        Ok(Self { ptr: layout })
    }
}

#[cfg(test)]
mod tests {
    use crate::LoadingError;

    use super::*;

    #[test]
    fn test_new_layout() {
        openvino_sys::library::load()
            .map_err(LoadingError::SystemFailure)
            .unwrap();
        let layout_desc = "NCHW";
        let layout = Layout::new(layout_desc).unwrap();
        assert!(!layout.ptr.is_null());
    }
}
