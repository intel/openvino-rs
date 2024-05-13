use crate::{try_unsafe, util::Result, ElementType, PartialShape, Shape};
use openvino_sys::{
    ov_const_port_get_shape, ov_output_const_port_t, ov_partial_shape_t, ov_port_get_any_name,
    ov_port_get_element_type, ov_port_get_partial_shape, ov_rank_t, ov_shape_t,
};

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

    /// Get the data type of elements of the port.
    pub fn get_element_type(&self) -> Result<u32> {
        let mut element_type = ElementType::Undefined as u32;
        try_unsafe!(ov_port_get_element_type(
            self.instance,
            std::ptr::addr_of_mut!(element_type),
        ))?;
        Ok(element_type)
    }

    /// Get the shape of the port.
    pub fn get_shape(&self) -> Result<Shape> {
        let mut instance = ov_shape_t {
            rank: 0,
            dims: std::ptr::null_mut(),
        };
        try_unsafe!(ov_const_port_get_shape(
            self.instance,
            std::ptr::addr_of_mut!(instance),
        ))?;
        Ok(Shape::new_from_instance(instance))
    }

    /// Get the partial shape of the port.
    pub fn get_partial_shape(&self) -> Result<PartialShape> {
        let mut instance = ov_partial_shape_t {
            rank: ov_rank_t { min: 0, max: 0 },
            dims: std::ptr::null_mut(),
        };
        try_unsafe!(ov_port_get_partial_shape(
            self.instance,
            std::ptr::addr_of_mut!(instance),
        ))?;
        Ok(PartialShape::new_from_instance(instance))
    }
}
