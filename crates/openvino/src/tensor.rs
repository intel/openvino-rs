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

    /// Get the underlying data for the tensor.
    pub fn get_raw_data(&self) -> Result<&[u8]> {
        let mut buffer = std::ptr::null_mut();
        try_unsafe!(ov_tensor_data(self.ptr, std::ptr::addr_of_mut!(buffer)))?;
        let size = self.get_byte_size()?;
        let slice = unsafe { std::slice::from_raw_parts(buffer.cast::<u8>(), size) };
        Ok(slice)
    }

    /// Get a mutable reference to the underlying data for the tensor.
    pub fn get_raw_data_mut(&mut self) -> Result<&mut [u8]> {
        let mut buffer = std::ptr::null_mut();
        try_unsafe!(ov_tensor_data(self.ptr, std::ptr::addr_of_mut!(buffer)))?;
        let size = self.get_byte_size()?;
        let slice = unsafe { std::slice::from_raw_parts_mut(buffer.cast::<u8>(), size) };
        Ok(slice)
    }

    /// Get a `T`-casted slice of the underlying data for the tensor.
    ///
    /// # Panics
    ///
    /// This method will panic if it can't cast the data to `T` due to the type size or the
    /// underlying pointer's alignment.
    pub fn get_data<T>(&self) -> Result<&[T]> {
        let raw_data = self.get_raw_data()?;
        let len = get_safe_len::<T>(raw_data);
        let slice = unsafe { std::slice::from_raw_parts(raw_data.as_ptr().cast::<T>(), len) };
        Ok(slice)
    }

    /// Get a mutable `T`-casted slice of the underlying data for the tensor.
    ///
    /// # Panics
    ///
    /// This method will panic if it can't cast the data to `T` due to the type size or the
    /// underlying pointer's alignment.
    pub fn get_data_mut<T>(&mut self) -> Result<&mut [T]> {
        let raw_data = self.get_raw_data_mut()?;
        let len = get_safe_len::<T>(raw_data);
        let slice =
            unsafe { std::slice::from_raw_parts_mut(raw_data.as_mut_ptr().cast::<T>(), len) };
        Ok(slice)
    }
}

/// Convenience function for checking that we can cast `data` to a slice of `T`, returning the
/// length of that slice.
fn get_safe_len<T>(data: &[u8]) -> usize {
    if data.len() % std::mem::size_of::<T>() != 0 {
        panic!("data size is not a multiple of the size of `T`");
    }
    if data.as_ptr() as usize % std::mem::align_of::<T>() != 0 {
        panic!("raw data is not aligned to `T`'s alignment");
    }
    data.len() / std::mem::size_of::<T>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_tensor() {
        openvino_sys::library::load().unwrap();
        let shape = Shape::new(&vec![1, 3, 227, 227]).unwrap();
        let tensor = Tensor::new(ElementType::F32, &shape).unwrap();
        assert!(!tensor.ptr.is_null());
    }

    #[test]
    fn test_get_shape() {
        openvino_sys::library::load().unwrap();
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
        openvino_sys::library::load().unwrap();
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
        openvino_sys::library::load().unwrap();
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
        openvino_sys::library::load().unwrap();
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

    #[test]
    fn casting() {
        openvino_sys::library::load().unwrap();
        let shape = Shape::new(&vec![10, 10, 10]).unwrap();
        let tensor = Tensor::new(ElementType::F32, &shape).unwrap();
        let data = tensor.get_data::<f32>().unwrap();
        assert_eq!(data.len(), 10 * 10 * 10);
    }

    #[test]
    #[should_panic(expected = "data size is not a multiple of the size of `T`")]
    fn casting_check() {
        openvino_sys::library::load().unwrap();
        let shape = Shape::new(&vec![10, 10, 10]).unwrap();
        let tensor = Tensor::new(ElementType::F32, &shape).unwrap();
        #[allow(dead_code)]
        struct LargeOddType([u8; 1061]);
        tensor.get_data::<LargeOddType>().unwrap();
    }
}
