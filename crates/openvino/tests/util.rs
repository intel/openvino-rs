#![allow(dead_code)] // Not all functions are used by each test.

use core::cmp::Ordering;
use float_cmp::{ApproxEq, F32Margin};
use openvino::version;

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

    /// Reduce the boilerplate to assert that two predictions are approximately the same.
    pub fn assert_approx_eq<P: Into<Self>>(&self, expected: P) {
        let expected = expected.into();
        assert_eq!(
            self.id, expected.id,
            "Expected class ID {} but found {}",
            expected.id, self.id
        );
        let approx_matches = self.approx_eq(&expected, DEFAULT_MARGIN);
        assert!(
            approx_matches,
            "Expected probability {} but found {} (outside of default margin of error)",
            expected.prob, self.prob
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

impl ApproxEq for &Prediction {
    type Margin = F32Margin;
    fn approx_eq<T: Into<Self::Margin>>(self, other: Self, margin: T) -> bool {
        let margin = margin.into();
        self.prob.approx_eq(other.prob, margin)
    }
}

/// The default margin for error allowed for comparing classification results.
pub const DEFAULT_MARGIN: F32Margin = F32Margin {
    epsilon: 0.01,
    ulps: 2,
};

/// A helper type for manipulating lists of results.
pub type Predictions = Vec<Prediction>;

/// OpenVINO's v2024.2 release introduced breaking changes to the C headers, upon which this crate
/// relies. This function checks if the running OpenVINO version is pre-2024.2.
pub fn is_version_pre_2024_2() -> bool {
    let version = version();
    let mut parts = version.parts();
    let year: usize = parts.next().unwrap().parse().unwrap();
    let minor: usize = parts.next().unwrap().parse().unwrap();
    year <= 2024 && minor < 2
}
