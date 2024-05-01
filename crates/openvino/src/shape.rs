use crate::{try_unsafe, util::Result};
use openvino_sys::{ov_shape_create, ov_shape_free, ov_shape_t};
use std::convert::TryInto;

/// See [`Shape`](https://docs.openvino.ai/2023.3/api/c_cpp_api/group__ov__shape__c__api.html).
pub struct Shape {
    instance: ov_shape_t,
}

impl Drop for Shape {
    /// Drops the Shape instance and frees the associated memory.
    //Not using drop! macro since ov_shape_free returns an error code unlike other free methods.
    fn drop(&mut self) {
        let code = unsafe { ov_shape_free(std::ptr::addr_of_mut!(self.instance)) };
        assert_eq!(code, 0);
        debug_assert!(self.instance.dims.is_null());
        debug_assert_eq!(self.instance.rank, 0);
    }
}

impl Shape {
    /// Get the pointer to the underlying OpenVINO shape.
    pub(crate) fn instance(&self) -> ov_shape_t {
        self.instance
    }

    /// Creates a new Shape instance with the given dimensions.
    pub fn new(dimensions: &[i64]) -> Result<Self> {
        let mut shape = ov_shape_t {
            rank: 8,
            dims: std::ptr::null_mut(),
        };
        try_unsafe!(ov_shape_create(
            dimensions.len().try_into().unwrap(),
            dimensions.as_ptr(),
            std::ptr::addr_of_mut!(shape)
        ))?;
        Ok(Self { instance: shape })
    }

    /// Create a new shape object from ov_shape_t.
    pub(crate) fn new_from_instance(instance: ov_shape_t) -> Self {
        Self { instance }
    }

    /// Returns the rank of the shape.
    pub fn get_rank(&self) -> i64 {
        self.instance.rank
    }
}

#[cfg(test)]
mod tests {
    use crate::LoadingError;

    use super::*;

    #[test]
    fn test_new_shape() {
        openvino_sys::library::load()
            .map_err(LoadingError::SystemFailure)
            .unwrap();
        let dimensions = vec![1, 2, 3, 4];
        let shape = Shape::new(&dimensions).unwrap();
        assert_eq!(shape.get_rank(), 4);
    }
}
