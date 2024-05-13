use openvino_sys::{ov_dimension_is_dynamic, ov_dimension_t};

/// See [`Dimension`](https://docs.openvino.ai/2023.3/api/c_cpp_api/group__ov__dimension__c__api.html).
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct Dimension {
    instance: ov_dimension_t,
}

impl PartialEq for Dimension {
    fn eq(&self, other: &Self) -> bool {
        self.instance.min == other.instance.min && self.instance.max == other.instance.max
    }
}

impl Eq for Dimension {}

impl Dimension {
    /// Get the pointer to the underlying OpenVINO dimension.
    #[allow(dead_code)]
    pub(crate) fn instance(&self) -> ov_dimension_t {
        self.instance
    }

    /// Create a new dimension object from `ov_dimension_t`.
    #[allow(dead_code)]
    pub(crate) fn new_from_instance(instance: ov_dimension_t) -> Self {
        Self { instance }
    }

    /// Creates a new Dimension with minimum and maximum values.
    pub fn new(min: i64, max: i64) -> Self {
        let instance = ov_dimension_t { min, max };
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

    /// Returns `true` if the dimension is dynamic.
    pub fn is_dynamic(&self) -> bool {
        unsafe { ov_dimension_is_dynamic(self.instance) }
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
