use crate::{try_unsafe, util::Result, ElementType, Shape};
use openvino_sys::{
    ov_const_port_get_shape, ov_output_const_port_t, ov_port_get_any_name,
    ov_port_get_element_type, ov_shape_t,
};
use std::borrow::Cow;
use std::ffi::CStr;

/// A [`Port`] is an input or output to a model.
pub struct Port {
    pub(crate) instance: *mut ov_output_const_port_t,
}

impl Port {
    /// Get the tensor name of the port.
    pub fn name(&self) -> Result<Cow<str>> {
        let mut c_name = std::ptr::null_mut();
        try_unsafe!(ov_port_get_any_name(
            self.instance,
            std::ptr::addr_of_mut!(c_name)
        ))?;
        let rust_name = unsafe { CStr::from_ptr(c_name) }.to_string_lossy();
        Ok(rust_name)
    }

    /// The shape of the port.
    pub fn shape(&self) -> Result<Shape> {
        let mut shape = ov_shape_t {
            rank: 0,
            dims: std::ptr::null_mut(),
        };
        try_unsafe!(ov_const_port_get_shape(
            self.instance,
            std::ptr::addr_of_mut!(shape)
        ))?;
        Ok(Shape { instance: shape })
    }

    /// The tensor data type of the port.
    pub fn element_type(&self) -> Result<ElementType> {
        let mut element_type: u32 = 0;
        try_unsafe!(ov_port_get_element_type(
            self.instance,
            std::ptr::addr_of_mut!(element_type),
        ))?;
        Ok(element_type.into())
    }
}
