use openvino_sys::{ov_rank_is_dynamic, ov_rank_t};

/// See [`ov_rank_t`](https://docs.openvino.ai/2024/api/c_cpp_api/group__ov__rank__c__api.html).
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct Rank {
    c_struct: ov_rank_t,
}

impl PartialEq for Rank {
    fn eq(&self, other: &Self) -> bool {
        self.c_struct.min == other.c_struct.min && self.c_struct.max == other.c_struct.max
    }
}

impl Eq for Rank {}

impl Rank {
    /// Get the pointer to the underlying OpenVINO rank.
    #[inline]
    pub(crate) fn as_c_struct(&self) -> ov_rank_t {
        self.c_struct
    }

    /// Create a new rank object from `ov_rank_t`.
    #[inline]
    pub(crate) fn from_c_struct(ptr: ov_rank_t) -> Self {
        Self { c_struct: ptr }
    }

    /// Creates a new Rank with minimum and maximum values.
    #[inline]
    pub fn new(min: i64, max: i64) -> Self {
        Self {
            c_struct: ov_rank_t { min, max },
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

    /// Returns `true` if the rank is dynamic.
    pub fn is_dynamic(&self) -> bool {
        unsafe { ov_rank_is_dynamic(self.c_struct) }
    }
}

#[cfg(test)]
mod tests {
    use crate::LoadingError;

    use super::Rank;

    #[test]
    fn test_static() {
        openvino_sys::library::load()
            .map_err(LoadingError::SystemFailure)
            .unwrap();

        let rank = Rank::new(1, 1);
        assert!(!rank.is_dynamic());
    }

    #[test]
    fn test_dynamic() {
        openvino_sys::library::load()
            .map_err(LoadingError::SystemFailure)
            .unwrap();

        let rank = Rank::new(1, 2);
        assert!(rank.is_dynamic());
    }
}
