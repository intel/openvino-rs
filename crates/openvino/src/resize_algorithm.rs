/// Interpolation mode when resizing during preprocess steps.
#[derive(Copy, Clone, Debug)]
#[repr(u32)]
pub enum ResizeAlgorithm {
    /// Linear interpolation
    Linear = openvino_sys::ov_preprocess_resize_algorithm_e_RESIZE_LINEAR,
    /// Cubic interpolation
    Cubic = openvino_sys::ov_preprocess_resize_algorithm_e_RESIZE_CUBIC,
    /// Nearest neighbor interpolation
    Nearest = openvino_sys::ov_preprocess_resize_algorithm_e_RESIZE_NEAREST,
}
