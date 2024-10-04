use crate::{try_unsafe, util::Result};
use openvino_sys::{ov_shape_create, ov_shape_free, ov_shape_t, ov_status_e};
use std::convert::TryInto;

/// See [`ov_shape_t`](https://docs.openvino.ai/2024/api/c_cpp_api/group__ov__shape__c__api.html).
pub struct Shape {
    c_struct: ov_shape_t,
}

impl Drop for Shape {
    // We don't use the `drop...!` macro here since:
    // - `ov_shape_free` returns an error code unlike other free methods
    // - the `c_struct` field is not a pointer as with other types.
    fn drop(&mut self) {
        let code = unsafe { ov_shape_free(std::ptr::addr_of_mut!(self.c_struct)) };
        assert_eq!(code, ov_status_e::OK);
        debug_assert!(self.c_struct.dims.is_null());
        debug_assert_eq!(self.c_struct.rank, 0);
    }
}

impl Shape {
    /// Creates a new [`Shape`] with the given dimensions.
    ///
    /// # Panics
    ///
    /// Panics in the unlikely case the dimension length cannot be represented as an `i64`.
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
        Ok(Self { c_struct: shape })
    }

    /// Create a new shape object from `ov_shape_t`.
    #[inline]
    pub(crate) fn from_c_struct(ptr: ov_shape_t) -> Self {
        Self { c_struct: ptr }
    }

    /// Get the pointer to the underlying OpenVINO shape.
    #[inline]
    pub(crate) fn as_c_struct(&self) -> ov_shape_t {
        self.c_struct
    }

    /// Returns the rank of the shape.
    #[inline]
    pub fn get_rank(&self) -> i64 {
        self.c_struct.rank
    }

    /// Returns the dimensions of the shape.
    ///
    /// # Panics
    ///
    /// Panics in the unlikely case the rank cannot be represented as a `usize`.
    pub fn get_dimensions(&self) -> &[i64] {
        if self.c_struct.dims.is_null() || self.c_struct.rank <= 0 {
            &[]
        } else {
            unsafe {
                std::slice::from_raw_parts(
                    self.c_struct.dims,
                    self.c_struct.rank.try_into().unwrap(),
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LoadingError;

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
