use openvino_sys::ov_preprocess_resize_algorithm_e;

/// Interpolation mode when resizing during preprocess steps.
#[derive(Copy, Clone, Debug)]
#[repr(u32)]
pub enum ResizeAlgorithm {
    /// Linear interpolation
    Linear,
    /// Cubic interpolation
    Cubic,
    /// Nearest neighbor interpolation
    Nearest,
}

impl From<ov_preprocess_resize_algorithm_e> for ResizeAlgorithm {
    fn from(algo: ov_preprocess_resize_algorithm_e) -> Self {
        match algo {
            ov_preprocess_resize_algorithm_e::RESIZE_LINEAR => Self::Linear,
            ov_preprocess_resize_algorithm_e::RESIZE_CUBIC => Self::Cubic,
            ov_preprocess_resize_algorithm_e::RESIZE_NEAREST => Self::Nearest,
        }
    }
}

impl From<ResizeAlgorithm> for ov_preprocess_resize_algorithm_e {
    fn from(algo: ResizeAlgorithm) -> ov_preprocess_resize_algorithm_e {
        match algo {
            ResizeAlgorithm::Linear => ov_preprocess_resize_algorithm_e::RESIZE_LINEAR,
            ResizeAlgorithm::Cubic => ov_preprocess_resize_algorithm_e::RESIZE_CUBIC,
            ResizeAlgorithm::Nearest => ov_preprocess_resize_algorithm_e::RESIZE_NEAREST,
        }
    }
}
