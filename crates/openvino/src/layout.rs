use crate::{drop_using_function, try_unsafe, util::Result};
use openvino_sys::{ov_layout_create, ov_layout_free, ov_layout_t};
use std::ffi::CString;

/// Represents a layout.
pub struct Layout {
    pub(crate) instance: *mut ov_layout_t,
}
drop_using_function!(Layout, ov_layout_free);

impl Layout {
    /// Creates a new layout with the given description.
    ///
    /// # Arguments
    ///
    /// * `layout_desc` - The description of the layout.
    ///
    /// # Returns
    ///
    /// A new `Layout` instance.
    pub fn new(layout_desc: &str) -> Result<Self> {
        let mut layout = std::ptr::null_mut();
        let c_layout_desc = CString::new(layout_desc).unwrap();
        try_unsafe!(ov_layout_create(
            c_layout_desc.as_ptr(),
            std::ptr::addr_of_mut!(layout)
        ))?;
        Ok(Self { instance: layout })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_layout() {
        let layout_desc = "NCHW";
        let layout = Layout::new(layout_desc).unwrap();
        assert_eq!(layout.instance.is_null(), false);
    }
}
