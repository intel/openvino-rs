use crate::{dimension::Dimension, try_unsafe, util::Result, Rank};
use openvino_sys::{
    ov_dimension_t, ov_partial_shape_create, ov_partial_shape_create_dynamic,
    ov_partial_shape_create_static, ov_partial_shape_free, ov_partial_shape_is_dynamic,
    ov_partial_shape_t, ov_rank_t,
};

use std::convert::TryInto;

/// See
/// [`ov_partial_shape_t`](https://docs.openvino.ai/2024/api/c_cpp_api/group__ov__partial__shape__c__api.html).
pub struct PartialShape {
    c_struct: ov_partial_shape_t,
}

impl Drop for PartialShape {
    // We don't use the `drop...!` macro here since:
    // - the `c_struct` field is not a pointer as with other types.
    fn drop(&mut self) {
        unsafe { ov_partial_shape_free(std::ptr::addr_of_mut!(self.c_struct)) }
    }
}

impl PartialShape {
    /// Create a new partial shape object from `ov_partial_shape_t`.
    #[inline]
    pub(crate) fn from_c_struct(c_struct: ov_partial_shape_t) -> Self {
        Self { c_struct }
    }

    /// Create a new [`PartialShape`] with a static rank and dynamic dimensions.
    pub fn new(rank: i64, dimensions: &[Dimension]) -> Result<Self> {
        let mut partial_shape = ov_partial_shape_t {
            rank: ov_rank_t { min: 0, max: 0 },
            dims: std::ptr::null_mut(),
        };
        try_unsafe!(ov_partial_shape_create(
            rank,
            dimensions.as_ptr().cast::<ov_dimension_t>(),
            std::ptr::addr_of_mut!(partial_shape)
        ))?;
        Ok(Self {
            c_struct: partial_shape,
        })
    }

    /// Create a new [`PartialShape`] with a dynamic rank and dynamic dimensions.
    pub fn new_dynamic(rank: Rank, dimensions: &[Dimension]) -> Result<Self> {
        let mut partial_shape = ov_partial_shape_t {
            rank: ov_rank_t { min: 0, max: 0 },
            dims: std::ptr::null_mut(),
        };
        try_unsafe!(ov_partial_shape_create_dynamic(
            rank.as_c_struct(),
            dimensions.as_ptr().cast::<ov_dimension_t>(),
            std::ptr::addr_of_mut!(partial_shape)
        ))?;
        Ok(Self {
            c_struct: partial_shape,
        })
    }

    /// Create a new [`PartialShape`] with a static rank and static dimensions.
    pub fn new_static(rank: i64, dimensions: &[i64]) -> Result<Self> {
        let mut partial_shape = ov_partial_shape_t {
            rank: ov_rank_t { min: 0, max: 0 },
            dims: std::ptr::null_mut(),
        };
        try_unsafe!(ov_partial_shape_create_static(
            rank,
            dimensions.as_ptr(),
            std::ptr::addr_of_mut!(partial_shape)
        ))?;
        Ok(Self {
            c_struct: partial_shape,
        })
    }

    /// Returns the rank of the partial shape.
    pub fn get_rank(&self) -> Rank {
        let rank = self.c_struct.rank;
        Rank::from_c_struct(rank)
    }

    /// Returns the dimensions of the partial shape.
    ///
    /// # Panics
    ///
    /// Panics in the unlikely case the rank cannot be represented as a `usize`.
    pub fn get_dimensions(&self) -> &[Dimension] {
        if self.c_struct.dims.is_null() {
            &[]
        } else {
            unsafe {
                std::slice::from_raw_parts(
                    self.c_struct.dims.cast::<Dimension>(),
                    self.c_struct.rank.max.try_into().unwrap(),
                )
            }
        }
    }

    /// Returns `true` if the partial shape is dynamic.
    pub fn is_dynamic(&self) -> bool {
        unsafe { ov_partial_shape_is_dynamic(self.c_struct) }
    }
}

#[cfg(test)]
mod tests {
    use crate::LoadingError;

    use super::*;

    #[test]
    fn test_new_partial_shape() {
        openvino_sys::library::load()
            .map_err(LoadingError::SystemFailure)
            .unwrap();

        let dimensions = vec![
            Dimension::new(0, 1),
            Dimension::new(1, 2),
            Dimension::new(2, 3),
            Dimension::new(3, 4),
        ];

        let shape = PartialShape::new(4, &dimensions).unwrap();
        assert_eq!(shape.get_rank().get_min(), 4);
        assert_eq!(shape.get_rank().get_max(), 4);
        assert!(shape.is_dynamic());
    }

    #[test]
    fn test_new_dynamic_partial_shape() {
        openvino_sys::library::load()
            .map_err(LoadingError::SystemFailure)
            .unwrap();

        let dimensions = vec![Dimension::new(1, 1), Dimension::new(2, 2)];

        let shape = PartialShape::new_dynamic(Rank::new(0, 2), &dimensions).unwrap();
        assert!(shape.is_dynamic());
    }

    #[test]
    fn test_new_static_partial_shape() {
        openvino_sys::library::load()
            .map_err(LoadingError::SystemFailure)
            .unwrap();

        let dimensions = vec![1, 2];

        let shape = PartialShape::new_static(2, &dimensions).unwrap();
        assert!(!shape.is_dynamic());
    }

    #[test]
    fn test_get_dimensions() {
        openvino_sys::library::load()
            .map_err(LoadingError::SystemFailure)
            .unwrap();

        let dimensions = vec![
            Dimension::new(0, 1),
            Dimension::new(1, 2),
            Dimension::new(2, 3),
            Dimension::new(3, 4),
        ];

        let shape = PartialShape::new(4, &dimensions).unwrap();

        let dims = shape.get_dimensions();

        assert_eq!(dims, &dimensions);
    }
}
