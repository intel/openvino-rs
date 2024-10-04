use crate::{try_unsafe, util::Result, ElementType, PartialShape, Shape};
use openvino_sys::{
    ov_const_port_get_shape, ov_element_type_e, ov_output_const_port_t, ov_partial_shape_t,
    ov_port_get_any_name, ov_port_get_element_type, ov_port_get_partial_shape, ov_rank_t,
    ov_shape_t,
};
use std::ffi::CStr;

/// See [`ov_node_c_api`](https://docs.openvino.ai/2024/api/c_cpp_api/group__ov__node__c__api.html).
pub struct Node {
    ptr: *mut ov_output_const_port_t,
}

impl Node {
    /// Create a new [`Port`] from [`ov_output_const_port_t`].
    #[inline]
    pub(crate) fn from_ptr(ptr: *mut ov_output_const_port_t) -> Self {
        Self { ptr }
    }

    /// Get name of a port.
    pub fn get_name(&self) -> Result<String> {
        let mut c_name = std::ptr::null_mut();
        try_unsafe!(ov_port_get_any_name(
            self.ptr,
            std::ptr::addr_of_mut!(c_name)
        ))?;
        let rust_name = unsafe { CStr::from_ptr(c_name) }
            .to_string_lossy()
            .into_owned();
        Ok(rust_name)
    }

    /// Get the data type of elements of the port.
    ///
    /// # Panics
    ///
    /// This function panics in the unlikely case OpenVINO returns an unknown element type.
    pub fn get_element_type(&self) -> Result<ElementType> {
        let mut element_type = ov_element_type_e::UNDEFINED;
        try_unsafe!(ov_port_get_element_type(
            self.ptr,
            std::ptr::addr_of_mut!(element_type),
        ))?;
        Ok(element_type.into())
    }

    /// Get the shape of the port.
    pub fn get_shape(&self) -> Result<Shape> {
        let mut shape = ov_shape_t {
            rank: 0,
            dims: std::ptr::null_mut(),
        };
        try_unsafe!(ov_const_port_get_shape(
            self.ptr,
            std::ptr::addr_of_mut!(shape),
        ))?;
        Ok(Shape::from_c_struct(shape))
    }

    /// Get the partial shape of the port.
    pub fn get_partial_shape(&self) -> Result<PartialShape> {
        let mut shape = ov_partial_shape_t {
            rank: ov_rank_t { min: 0, max: 0 },
            dims: std::ptr::null_mut(),
        };
        try_unsafe!(ov_port_get_partial_shape(
            self.ptr,
            std::ptr::addr_of_mut!(shape),
        ))?;
        Ok(PartialShape::from_c_struct(shape))
    }
}
