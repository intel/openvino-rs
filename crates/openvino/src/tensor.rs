//! This module provides functionality related to Tensor objects.
use std::convert::TryInto as _;

use crate::element_type::ElementType;
use crate::shape::Shape;
use crate::{drop_using_function, try_unsafe, util::Result};
use openvino_sys::{
    self, ov_shape_t, ov_tensor_create, ov_tensor_data, ov_tensor_free, ov_tensor_get_byte_size,
    ov_tensor_get_element_type, ov_tensor_get_shape, ov_tensor_get_size, ov_tensor_set_shape,
    ov_tensor_t,
};

/// See [`Tensor`](https://docs.openvino.ai/2023.3/api/c_cpp_api/group__ov__tensor__c__api.html).
pub struct Tensor {
    ptr: *mut ov_tensor_t,
}
drop_using_function!(Tensor, ov_tensor_free);

impl Tensor {
    /// Create a new [`Tensor`].
    pub fn new(element_type: ElementType, shape: &Shape) -> Result<Self> {
        let mut ptr = std::ptr::null_mut();
        try_unsafe!(ov_tensor_create(
            element_type as u32,
            shape.as_c_struct(),
            std::ptr::addr_of_mut!(ptr),
        ))?;
        Ok(Self { ptr })
    }

    /// Create a new [`Tensor`] from a pointer.
    #[inline]
    pub(crate) fn from_ptr(ptr: *mut ov_tensor_t) -> Self {
        Self { ptr }
    }

    /// Get the pointer to the underlying OpenVINO tensor.
    #[inline]
    pub(crate) fn as_ptr(&self) -> *const ov_tensor_t {
        self.ptr
    }

    /// (Re)Set the shape of the tensor to a new shape.
    pub fn set_shape(&self, shape: Shape) -> Result<Self> {
        try_unsafe!(ov_tensor_set_shape(self.ptr, shape.as_c_struct()))?;
        Ok(Self { ptr: self.ptr })
    }

    /// Get the shape of the tensor.
    pub fn get_shape(&self) -> Result<Shape> {
        let mut shape = ov_shape_t {
            rank: 0,
            dims: std::ptr::null_mut(),
        };
        try_unsafe!(ov_tensor_get_shape(self.ptr, std::ptr::addr_of_mut!(shape),))?;
        Ok(Shape::from_c_struct(shape))
    }

    /// Get the data type of elements of the tensor.
    pub fn get_element_type(&self) -> Result<ElementType> {
        let mut element_type = ElementType::Undefined as u32;
        try_unsafe!(ov_tensor_get_element_type(
            self.ptr,
            std::ptr::addr_of_mut!(element_type),
        ))?;
        Ok(element_type.try_into().unwrap())
    }

    /// Get the number of elements in the tensor. Product of all dimensions e.g. 1*3*227*227.
    pub fn get_size(&self) -> Result<usize> {
        let mut elements_size = 0;
        try_unsafe!(ov_tensor_get_size(
            self.ptr,
            std::ptr::addr_of_mut!(elements_size)
        ))?;
        Ok(elements_size)
    }

    /// Get the size of the tensor in bytes.
    pub fn get_byte_size(&self) -> Result<usize> {
        let mut byte_size: usize = 0;
        try_unsafe!(ov_tensor_get_byte_size(
            self.ptr,
            std::ptr::addr_of_mut!(byte_size),
        ))?;
        Ok(byte_size)
    }

    /// Get a mutable reference to the data of the tensor.
    pub fn get_data<T>(&mut self) -> Result<&mut [T]> {
        let mut data = std::ptr::null_mut();
        try_unsafe!(ov_tensor_data(self.ptr, std::ptr::addr_of_mut!(data),))?;
        let size = self.get_byte_size()? / std::mem::size_of::<T>();
        let slice = unsafe { std::slice::from_raw_parts_mut(data.cast::<T>(), size) };
        Ok(slice)
    }

    /// Get a mutable reference to the buffer of the tensor.
    ///
    /// # Returns
    ///
    /// A mutable reference to the buffer of the tensor.
    pub fn buffer_mut(&mut self) -> Result<&mut [u8]> {
        let mut buffer = std::ptr::null_mut();
        try_unsafe!(ov_tensor_data(self.ptr, std::ptr::addr_of_mut!(buffer)))?;
        let size = self.get_byte_size()?;
        let slice = unsafe { std::slice::from_raw_parts_mut(buffer.cast::<u8>(), size) };
        Ok(slice)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ElementType, LoadingError, Shape};

    #[test]
    fn test_create_tensor() {
        openvino_sys::library::load()
            .map_err(LoadingError::SystemFailure)
            .unwrap();
        let shape = Shape::new(&vec![1, 3, 227, 227]).unwrap();
        let tensor = Tensor::new(ElementType::F32, &shape).unwrap();
        assert!(!tensor.ptr.is_null());
    }

    #[test]
    fn test_get_shape() {
        openvino_sys::library::load()
            .map_err(LoadingError::SystemFailure)
            .unwrap();
        let tensor = Tensor::new(
            ElementType::F32,
            &Shape::new(&vec![1, 3, 227, 227]).unwrap(),
        )
        .unwrap();
        let shape = tensor.get_shape().unwrap();
        assert_eq!(shape.get_rank(), 4);
    }

    #[test]
    fn test_get_element_type() {
        openvino_sys::library::load()
            .map_err(LoadingError::SystemFailure)
            .unwrap();
        let tensor = Tensor::new(
            ElementType::F32,
            &Shape::new(&vec![1, 3, 227, 227]).unwrap(),
        )
        .unwrap();
        let element_type = tensor.get_element_type().unwrap();
        assert_eq!(element_type, ElementType::F32);
    }

    #[test]
    fn test_get_size() {
        openvino_sys::library::load()
            .map_err(LoadingError::SystemFailure)
            .unwrap();
        let tensor = Tensor::new(
            ElementType::F32,
            &Shape::new(&vec![1, 3, 227, 227]).unwrap(),
        )
        .unwrap();
        let size = tensor.get_size().unwrap();
        assert_eq!(size, 1 * 3 * 227 * 227);
    }

    #[test]
    fn test_get_byte_size() {
        openvino_sys::library::load()
            .map_err(LoadingError::SystemFailure)
            .unwrap();
        let tensor = Tensor::new(
            ElementType::F32,
            &Shape::new(&vec![1, 3, 227, 227]).unwrap(),
        )
        .unwrap();
        let byte_size = tensor.get_byte_size().unwrap();
        assert_eq!(
            byte_size,
            1 * 3 * 227 * 227 * std::mem::size_of::<f32>() as usize
        );
    }
}
