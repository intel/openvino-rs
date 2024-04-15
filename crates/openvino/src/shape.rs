use crate::{try_unsafe, util::Result};
use openvino_sys::{ov_shape_create, ov_shape_free, ov_shape_t};
use std::convert::TryInto;
use std::slice;

/// Represents a shape in OpenVINO.
pub struct Shape {
    pub(crate) instance: ov_shape_t,
}

impl Drop for Shape {
    /// Drops the Shape instance and frees the associated memory.
    //Not using drop! macro since ov_shape_free returns an error code unlike other free methods
    fn drop(&mut self) {
        let code = unsafe { ov_shape_free(std::ptr::addr_of_mut!(self.instance)) };
        assert_eq!(code, 0);
        debug_assert!(self.instance.dims.is_null());
        debug_assert_eq!(self.instance.rank, 0);
    }
}

impl Shape {
    /// Creates a new Shape instance with the given dimensions.
    ///
    /// # Arguments
    ///
    /// * `dimensions` - A vector of dimensions for the shape.
    pub fn new(dimensions: &[i64]) -> Result<Self> {
        let mut shape = ov_shape_t {
            rank: 0,
            dims: std::ptr::null_mut(),
        };
        try_unsafe!(ov_shape_create(
            dimensions.len().try_into().unwrap(),
            dimensions.as_ptr(),
            std::ptr::addr_of_mut!(shape)
        ))?;
        Ok(Self { instance: shape })
    }

    /// The rank of the shape (i.e. the number of dimensions).
    pub fn rank(&self) -> i64 {
        self.instance.rank
    }

    /// The dimensions of the shape.
    pub fn dims(&self) -> &[i64] {
        unsafe { slice::from_raw_parts(self.instance.dims, self.instance.rank as usize) }
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
        assert_eq!(shape.rank(), 4);
    }
}
