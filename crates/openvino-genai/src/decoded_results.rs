//! Safe wrapper for [`ov_genai_decoded_results`].

use crate::{drop_using_function, util::Result, PerfMetrics};
use openvino_genai_sys::{
    self, ov_genai_decoded_results, ov_genai_decoded_results_free,
    ov_genai_decoded_results_get_perf_metrics, ov_genai_decoded_results_get_string,
};

/// Results from LLM text generation.
pub struct DecodedResults {
    pub(crate) ptr: *mut ov_genai_decoded_results,
}
drop_using_function!(DecodedResults, ov_genai_decoded_results_free);

impl DecodedResults {
    /// Get the generated text as a string.
    pub fn get_string(&self) -> Result<String> {
        unsafe {
            crate::util::get_c_string_two_call(|buf, size| {
                ov_genai_decoded_results_get_string(self.ptr, buf, size)
            })
        }
    }

    /// Get performance metrics from these results.
    pub fn get_perf_metrics(&self) -> Result<PerfMetrics> {
        let mut ptr = std::ptr::null_mut();
        crate::InferenceError::convert(unsafe {
            ov_genai_decoded_results_get_perf_metrics(self.ptr, std::ptr::addr_of_mut!(ptr))
        })?;
        Ok(PerfMetrics::from_decoded_results(ptr))
    }

    /// Construct from a raw pointer. For internal use.
    pub(crate) fn from_ptr(ptr: *mut ov_genai_decoded_results) -> Self {
        Self { ptr }
    }
}
