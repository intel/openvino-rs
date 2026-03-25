//! Safe wrapper for [`ov_genai_perf_metrics`].

use crate::{try_unsafe, util::Result};
use openvino_genai_sys::{
    self, ov_genai_perf_metrics, ov_genai_perf_metrics_get_generate_duration,
    ov_genai_perf_metrics_get_load_time, ov_genai_perf_metrics_get_num_generation_tokens,
    ov_genai_perf_metrics_get_num_input_tokens, ov_genai_perf_metrics_get_throughput,
    ov_genai_perf_metrics_get_tpot, ov_genai_perf_metrics_get_ttft,
};

/// Which free function to use depends on how the metrics were obtained.
#[derive(Debug, Clone, Copy)]
enum MetricsSource {
    Llm,
    Vlm,
    Whisper,
}

/// Performance metrics from a generation operation.
///
/// Contains timing information such as time-to-first-token, tokens per second, etc.
pub struct PerfMetrics {
    ptr: *mut ov_genai_perf_metrics,
    source: MetricsSource,
}

impl Drop for PerfMetrics {
    fn drop(&mut self) {
        unsafe {
            match self.source {
                MetricsSource::Llm => {
                    openvino_genai_sys::ov_genai_decoded_results_perf_metrics_free(self.ptr);
                }
                MetricsSource::Vlm => {
                    openvino_genai_sys::ov_genai_vlm_decoded_results_perf_metrics_free(self.ptr);
                }
                MetricsSource::Whisper => {
                    // Whisper decoded results don't have a separate perf_metrics_free;
                    // the metrics are freed when the results are freed. This is a no-op.
                }
            }
        }
    }
}

impl PerfMetrics {
    /// Get the model load time in milliseconds.
    pub fn get_load_time(&self) -> Result<f32> {
        let mut value: f32 = 0.0;
        try_unsafe!(ov_genai_perf_metrics_get_load_time(self.ptr, &mut value))?;
        Ok(value)
    }

    /// Get the number of generated tokens.
    pub fn get_num_generation_tokens(&self) -> Result<usize> {
        let mut value: usize = 0;
        try_unsafe!(ov_genai_perf_metrics_get_num_generation_tokens(
            self.ptr, &mut value
        ))?;
        Ok(value)
    }

    /// Get the number of input tokens.
    pub fn get_num_input_tokens(&self) -> Result<usize> {
        let mut value: usize = 0;
        try_unsafe!(ov_genai_perf_metrics_get_num_input_tokens(
            self.ptr, &mut value
        ))?;
        Ok(value)
    }

    /// Get the time to first token (mean, std) in milliseconds.
    pub fn get_ttft(&self) -> Result<(f32, f32)> {
        let mut mean: f32 = 0.0;
        let mut std: f32 = 0.0;
        try_unsafe!(ov_genai_perf_metrics_get_ttft(
            self.ptr, &mut mean, &mut std
        ))?;
        Ok((mean, std))
    }

    /// Get the time per output token (mean, std) in milliseconds.
    pub fn get_tpot(&self) -> Result<(f32, f32)> {
        let mut mean: f32 = 0.0;
        let mut std: f32 = 0.0;
        try_unsafe!(ov_genai_perf_metrics_get_tpot(
            self.ptr, &mut mean, &mut std
        ))?;
        Ok((mean, std))
    }

    /// Get throughput in tokens per second (mean, std).
    pub fn get_throughput(&self) -> Result<(f32, f32)> {
        let mut mean: f32 = 0.0;
        let mut std: f32 = 0.0;
        try_unsafe!(ov_genai_perf_metrics_get_throughput(
            self.ptr, &mut mean, &mut std
        ))?;
        Ok((mean, std))
    }

    /// Get the total generation duration (mean, std) in milliseconds.
    pub fn get_generate_duration(&self) -> Result<(f32, f32)> {
        let mut mean: f32 = 0.0;
        let mut std: f32 = 0.0;
        try_unsafe!(ov_genai_perf_metrics_get_generate_duration(
            self.ptr, &mut mean, &mut std
        ))?;
        Ok((mean, std))
    }

    /// Construct from a decoded results perf metrics pointer.
    pub(crate) fn from_decoded_results(ptr: *mut ov_genai_perf_metrics) -> Self {
        Self {
            ptr,
            source: MetricsSource::Llm,
        }
    }

    /// Construct from a VLM decoded results perf metrics pointer.
    pub(crate) fn from_vlm_decoded_results(ptr: *mut ov_genai_perf_metrics) -> Self {
        Self {
            ptr,
            source: MetricsSource::Vlm,
        }
    }

    /// Construct from a Whisper decoded results perf metrics pointer.
    pub(crate) fn from_whisper_decoded_results(ptr: *mut ov_genai_perf_metrics) -> Self {
        Self {
            ptr,
            source: MetricsSource::Whisper,
        }
    }
}
