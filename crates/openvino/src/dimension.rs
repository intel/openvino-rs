use openvino_sys::{ov_dimension_is_dynamic, ov_dimension_t};

/// See
/// [`ov_dimension_t`](https://docs.openvino.ai/2024/api/c_cpp_api/group__ov__dimension__c__api.html).
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct Dimension {
    c_struct: ov_dimension_t,
}

impl PartialEq for Dimension {
    fn eq(&self, other: &Self) -> bool {
        self.c_struct.min == other.c_struct.min && self.c_struct.max == other.c_struct.max
    }
}

impl Eq for Dimension {}

impl Dimension {
    /// Creates a new Dimension with minimum and maximum values.
    #[inline]
    pub fn new(min: i64, max: i64) -> Self {
        Self {
            c_struct: ov_dimension_t { min, max },
        }
    }

    /// Returns the minimum value.
    #[inline]
    pub fn get_min(&self) -> i64 {
        self.c_struct.min
    }

    /// Returns the maximum value.
    #[inline]
    pub fn get_max(&self) -> i64 {
        self.c_struct.max
    }

    /// Returns `true` if the dimension is dynamic.
    pub fn is_dynamic(&self) -> bool {
        unsafe { ov_dimension_is_dynamic(self.c_struct) }
    }
}

#[cfg(test)]
mod tests {
    use crate::LoadingError;

    use super::Dimension;

    #[test]
    fn test_static() {
        openvino_sys::library::load()
            .map_err(LoadingError::SystemFailure)
            .unwrap();

        let dim = Dimension::new(1, 1);
        assert!(!dim.is_dynamic());
    }

    #[test]
    fn test_dynamic() {
        openvino_sys::library::load()
            .map_err(LoadingError::SystemFailure)
            .unwrap();

        let dim = Dimension::new(1, 2);
        assert!(dim.is_dynamic());
    }
}
