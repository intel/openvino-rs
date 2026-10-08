use core::cmp::Ordering;

/// A structure for holding the `(category, probability)` pair extracted from the output tensor of
/// the OpenVINO classification.
#[derive(Debug)]
pub struct Prediction {
    id: usize,
    prob: f32,
}

impl Prediction {
    pub fn new(id: usize, prob: f32) -> Self {
        Self { id, prob }
    }

    /// Assert that this prediction is for the expected class.
    ///
    /// Only the class ID is checked, and in practice this is only worth calling on `results[0]`.
    /// Neither the probabilities nor the ranking below the top result are stable: they shift with
    /// the OpenVINO release and with the machine the test runs on, because the selected kernels
    /// and the order they accumulate in both differ. The lower-ranked classes for these fixtures
    /// sit within a few thousandths of each other, so they reorder and even swap membership. Only
    /// the top-ranked class has held across every observed version and host.
    pub fn assert_class(&self, expected_id: usize) {
        assert_eq!(
            self.id, expected_id,
            "Expected class ID {} but found {} (probability {})",
            expected_id, self.id, self.prob
        );
    }
}

impl From<(usize, f32)> for Prediction {
    fn from(p: (usize, f32)) -> Self {
        Prediction::new(p.0, p.1)
    }
}

/// Classification results are ordered by their probability, from greatest to smallest.
impl Ord for Prediction {
    fn cmp(&self, other: &Self) -> Ordering {
        assert!(!self.prob.is_nan());
        assert!(!other.prob.is_nan());
        other
            .prob
            .partial_cmp(&self.prob)
            .expect("a comparable value")
    }
}

impl PartialOrd for Prediction {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Prediction {
    fn eq(&self, other: &Self) -> bool {
        self.prob == other.prob
    }
}

impl Eq for Prediction {}

/// A helper type for manipulating lists of results.
pub type Predictions = Vec<Prediction>;
