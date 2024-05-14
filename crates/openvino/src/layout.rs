use crate::{cstr, drop_using_function, try_unsafe, util::Result};
use openvino_sys::{ov_layout_create, ov_layout_free, ov_layout_t};

/// See [`Layout`](https://docs.openvino.ai/2023.3/api/c_cpp_api/group__ov__layout__c__api.html).
pub struct Layout {
    instance: *mut ov_layout_t,
}
drop_using_function!(Layout, ov_layout_free);

impl Layout {
    /// Get [`ov_layout_t`] instance.
    pub(crate) fn instance(&self) -> *mut ov_layout_t {
        self.instance
    }

    /// Creates a new layout with the given description.
    pub fn new(layout_desc: &str) -> Result<Self> {
        let mut layout = std::ptr::null_mut();
        try_unsafe!(ov_layout_create(
            cstr!(layout_desc),
            std::ptr::addr_of_mut!(layout)
        ))?;
        Ok(Self { instance: layout })
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
        assert_eq!(layout.instance.is_null(), false);
    }
}
