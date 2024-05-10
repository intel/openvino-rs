use openvino_sys::{ov_rank_is_dynamic, ov_rank_t};

/// See [`Rank`](https://docs.openvino.ai/2023.3/api/c_cpp_api/group__ov__rank__c__api.html).
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct Rank {
    instance: ov_rank_t,
}

impl PartialEq for Rank {
    fn eq(&self, other: &Self) -> bool {
        self.instance.min == other.instance.min && self.instance.max == other.instance.max
    }
}

impl Eq for Rank {}

impl Rank {
    /// Get the pointer to the underlying OpenVINO rank.
    pub(crate) fn instance(&self) -> ov_rank_t {
        self.instance
    }

    /// Create a new rank object from `ov_rank_t`.
    pub(crate) fn new_from_instance(instance: ov_rank_t) -> Self {
        Self { instance }
    }

    /// Creates a new Rank with minimum and maximum values.
    pub fn new(min: i64, max: i64) -> Self {
        let instance = ov_rank_t { min, max };
        Self { instance }
    }

    /// Returns the minimum value.
    pub fn get_min(&self) -> i64 {
        self.instance.min
    }

    /// Returns the maximum value.
    pub fn get_max(&self) -> i64 {
        self.instance.max
    }

    /// Returns `true` if the rank is dynamic.
    pub fn is_dynamic(&self) -> bool {
        unsafe { ov_rank_is_dynamic(self.instance) }
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
