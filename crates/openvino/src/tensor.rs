//! This module provides functionality related to Tensor objects.
use crate::element_type::ElementType;
use crate::shape::Shape;
use crate::{drop_using_function, try_unsafe, util::Result};
use openvino_sys::{
    self, ov_shape_t, ov_tensor_create, ov_tensor_create_from_host_ptr, ov_tensor_data,
    ov_tensor_free, ov_tensor_get_byte_size, ov_tensor_get_element_type, ov_tensor_get_shape,
    ov_tensor_get_size, ov_tensor_set_shape, ov_tensor_t,
};

/// See [`Tensor`](https://docs.openvino.ai/2023.3/api/c_cpp_api/group__ov__tensor__c__api.html).
pub struct Tensor {
    /// Pointer to the underlying OpenVINO tensor.
    pub(crate) instance: *mut ov_tensor_t,
}
drop_using_function!(Tensor, ov_tensor_free);

unsafe impl Send for Tensor {}

impl Tensor {
    /// Create a new [`Tensor`].
    ///
    /// # Arguments
    ///
    /// * `data_type` - The data type of the tensor.
    /// * `shape` - The shape of the tensor.
    ///
    /// # Returns
    ///
    /// A new `Tensor` object.
    pub fn new(data_type: ElementType, shape: &Shape) -> Result<Self> {
        let mut tensor = std::ptr::null_mut();
        let element_type = data_type as u32;
        let code = try_unsafe!(ov_tensor_create(
            element_type,
            shape.instance,
            std::ptr::addr_of_mut!(tensor),
        ));
        assert_eq!(code, Ok(()));
        Ok(Self { instance: tensor })
    }

    /// Create a new [`Tensor`] from a host pointer.
    ///
    /// # Arguments
    ///
    /// * `data_type` - The data type of the tensor.
    /// * `shape` - The shape of the tensor.
    /// * `data` - The data buffer.
    ///
    /// # Returns
    ///
    /// A new `Tensor` object.
    pub fn new_from_host_ptr(data_type: ElementType, shape: &Shape, data: &[u8]) -> Result<Self> {
        let mut tensor: *mut ov_tensor_t = std::ptr::null_mut();
        let element_type: u32 = data_type as u32;
        let buffer = data.as_ptr() as *mut std::os::raw::c_void;
        try_unsafe!(ov_tensor_create_from_host_ptr(
            element_type,
            shape.instance,
            buffer,
            std::ptr::addr_of_mut!(tensor)
        ))?;
        Ok(Self { instance: tensor })
    }

    /// (Re)Set the shape of the tensor to a new shape
    ///
    /// # Arguments
    ///
    /// * `shape` - The new shape of the tensor.
    ///
    /// # Returns
    ///
    /// A new `Tensor` object with the updated shape.
    pub fn set_shape(&self, shape: &Shape) -> Result<Self> {
        try_unsafe!(ov_tensor_set_shape(self.instance, shape.instance))?;
        Ok(Self {
            instance: self.instance,
        })
    }

    /// Get the shape of the tensor.
    ///
    /// # Returns
    ///
    /// The shape of the tensor.
    pub fn shape(&self) -> Result<Shape> {
        let mut instance = ov_shape_t {
            rank: 0,
            dims: std::ptr::null_mut(),
        };
        try_unsafe!(ov_tensor_get_shape(
            self.instance,
            std::ptr::addr_of_mut!(instance),
        ))?;
        Ok(Shape { instance })
    }

    /// Get the data type of elements of the tensor.
    ///
    /// # Returns
    ///
    /// The data type of elements of the tensor.
    pub fn element_type(&self) -> Result<ElementType> {
        let mut element_type: u32 = 0;
        try_unsafe!(ov_tensor_get_element_type(
            self.instance,
            std::ptr::addr_of_mut!(element_type),
        ))?;
        Ok(element_type.into())
    }

    /// Get the number of elements in the tensor. Product of all dimensions e.g. 1*3*227*227
    ///
    /// # Returns
    ///
    /// The number of elements in the tensor.
    pub fn size(&self) -> Result<usize> {
        let mut elements_size = 0;
        try_unsafe!(ov_tensor_get_size(
            self.instance,
            std::ptr::addr_of_mut!(elements_size)
        ))?;
        Ok(elements_size)
    }

    /// Get the size of the tensor in bytes.
    ///
    /// # Returns
    ///
    /// The size of the tensor in bytes.
    pub fn byte_size(&self) -> Result<usize> {
        let mut byte_size: usize = 0;
        try_unsafe!(ov_tensor_get_byte_size(
            self.instance,
            std::ptr::addr_of_mut!(byte_size),
        ))?;
        Ok(byte_size)
    }

    /// Get a mutable reference to the data of the tensor.
    ///
    /// # Returns
    ///
    /// A mutable reference to the data of the tensor.
    pub fn data<T>(&mut self) -> Result<&mut [T]> {
        let mut data = std::ptr::null_mut();
        try_unsafe!(ov_tensor_data(self.instance, std::ptr::addr_of_mut!(data),))?;
        let size = self.byte_size()? / std::mem::size_of::<T>();
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
        try_unsafe!(ov_tensor_data(
            self.instance,
            std::ptr::addr_of_mut!(buffer)
        ))?;
        let size = self.byte_size()?;
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
        assert!(!tensor.instance.is_null());
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
        let shape = tensor.shape().unwrap();
        assert_eq!(shape.rank(), 4);
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
        let element_type = tensor.element_type().unwrap();
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
        let size = tensor.size().unwrap();
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
        let byte_size = tensor.byte_size().unwrap();
        assert_eq!(
            byte_size,
            1 * 3 * 227 * 227 * std::mem::size_of::<f32>() as usize
        );
    }
}
