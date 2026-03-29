//! Safe wrapper for [`ov_genai_whisper_pipeline`] — Whisper speech recognition pipeline.

use crate::error::LoadingError;
use crate::{
    cstr, drop_using_function, try_unsafe, util::Result, PerfMetrics, SetupError,
    WhisperGenerationConfig,
};
use openvino_genai_sys::{
    self, ov_genai_whisper_decoded_result_chunk, ov_genai_whisper_decoded_result_chunk_free,
    ov_genai_whisper_decoded_result_chunk_get_end_ts,
    ov_genai_whisper_decoded_result_chunk_get_start_ts,
    ov_genai_whisper_decoded_result_chunk_get_text, ov_genai_whisper_decoded_results,
    ov_genai_whisper_decoded_results_free, ov_genai_whisper_decoded_results_get_chunk_at,
    ov_genai_whisper_decoded_results_get_chunks_count,
    ov_genai_whisper_decoded_results_get_perf_metrics,
    ov_genai_whisper_decoded_results_get_score_at, ov_genai_whisper_decoded_results_get_string,
    ov_genai_whisper_decoded_results_get_text_at, ov_genai_whisper_decoded_results_get_texts_count,
    ov_genai_whisper_decoded_results_has_chunks, ov_genai_whisper_pipeline,
    ov_genai_whisper_pipeline_create, ov_genai_whisper_pipeline_free,
    ov_genai_whisper_pipeline_generate, ov_genai_whisper_pipeline_get_generation_config,
    ov_genai_whisper_pipeline_set_generation_config,
};

/// A pipeline for speech recognition using Whisper models.
pub struct WhisperPipeline {
    ptr: *mut ov_genai_whisper_pipeline,
}
drop_using_function!(WhisperPipeline, ov_genai_whisper_pipeline_free);

unsafe impl Send for WhisperPipeline {}

impl WhisperPipeline {
    /// Create a new Whisper pipeline from a model directory and device name.
    pub fn new(models_path: &str, device: &str) -> std::result::Result<Self, SetupError> {
        openvino_genai_sys::library::load().map_err(LoadingError::SystemFailure)?;
        let models_path = cstr!(models_path);
        let device = cstr!(device);
        let mut ptr = std::ptr::null_mut();
        try_unsafe!(ov_genai_whisper_pipeline_create(
            models_path.as_ptr(),
            device.as_ptr(),
            0,
            std::ptr::addr_of_mut!(ptr),
            &[]
        ))?;
        Ok(Self { ptr })
    }

    /// Generate transcription from raw audio samples.
    ///
    /// The `audio` parameter should contain raw PCM float samples (typically 16kHz mono).
    pub fn generate(
        &mut self,
        audio: &[f32],
        config: Option<&WhisperGenerationConfig>,
    ) -> Result<WhisperDecodedResults> {
        let config_ptr = config.map_or(std::ptr::null(), WhisperGenerationConfig::as_ptr);
        let mut results_ptr = std::ptr::null_mut();
        try_unsafe!(ov_genai_whisper_pipeline_generate(
            self.ptr,
            audio.as_ptr(),
            audio.len(),
            config_ptr,
            std::ptr::addr_of_mut!(results_ptr)
        ))?;
        Ok(WhisperDecodedResults::from_ptr(results_ptr))
    }

    /// Get the pipeline's current whisper generation config.
    pub fn get_generation_config(&self) -> Result<WhisperGenerationConfig> {
        let mut ptr = std::ptr::null_mut();
        try_unsafe!(ov_genai_whisper_pipeline_get_generation_config(
            self.ptr,
            std::ptr::addr_of_mut!(ptr)
        ))?;
        Ok(WhisperGenerationConfig::from_ptr(ptr))
    }

    /// Set the whisper generation config for this pipeline.
    pub fn set_generation_config(&mut self, config: &mut WhisperGenerationConfig) -> Result<()> {
        try_unsafe!(ov_genai_whisper_pipeline_set_generation_config(
            self.ptr,
            config.as_mut_ptr()
        ))
    }
}

/// Results from Whisper speech recognition.
pub struct WhisperDecodedResults {
    ptr: *mut ov_genai_whisper_decoded_results,
}
drop_using_function!(WhisperDecodedResults, ov_genai_whisper_decoded_results_free);

impl WhisperDecodedResults {
    /// Get the full transcription as a single string.
    pub fn get_string(&self) -> Result<String> {
        unsafe {
            crate::util::get_c_string_two_call(|buf, size| {
                ov_genai_whisper_decoded_results_get_string(self.ptr, buf, size)
            })
        }
    }

    /// Get the number of text results.
    pub fn get_texts_count(&self) -> Result<usize> {
        let mut count: usize = 0;
        try_unsafe!(ov_genai_whisper_decoded_results_get_texts_count(
            self.ptr, &mut count
        ))?;
        Ok(count)
    }

    /// Get a specific text result by index.
    pub fn get_text_at(&self, index: usize) -> Result<String> {
        unsafe {
            crate::util::get_c_string_two_call(|buf, size| {
                ov_genai_whisper_decoded_results_get_text_at(self.ptr, index, buf, size)
            })
        }
    }

    /// Get the score at a specific index.
    pub fn get_score_at(&self, index: usize) -> Result<f32> {
        let mut score: f32 = 0.0;
        try_unsafe!(ov_genai_whisper_decoded_results_get_score_at(
            self.ptr, index, &mut score
        ))?;
        Ok(score)
    }

    /// Check if timestamp chunks are available.
    pub fn has_chunks(&self) -> Result<bool> {
        let mut has: bool = false;
        try_unsafe!(ov_genai_whisper_decoded_results_has_chunks(
            self.ptr, &mut has
        ))?;
        Ok(has)
    }

    /// Get the number of chunks.
    pub fn get_chunks_count(&self) -> Result<usize> {
        let mut count: usize = 0;
        try_unsafe!(ov_genai_whisper_decoded_results_get_chunks_count(
            self.ptr, &mut count
        ))?;
        Ok(count)
    }

    /// Get a specific chunk by index.
    pub fn get_chunk_at(&self, index: usize) -> Result<WhisperDecodedResultChunk> {
        let mut ptr = std::ptr::null_mut();
        try_unsafe!(ov_genai_whisper_decoded_results_get_chunk_at(
            self.ptr,
            index,
            std::ptr::addr_of_mut!(ptr)
        ))?;
        Ok(WhisperDecodedResultChunk::from_ptr(ptr))
    }

    /// Get performance metrics from these results.
    pub fn get_perf_metrics(&self) -> Result<PerfMetrics> {
        let mut ptr = std::ptr::null_mut();
        crate::InferenceError::convert(unsafe {
            ov_genai_whisper_decoded_results_get_perf_metrics(
                self.ptr,
                std::ptr::addr_of_mut!(ptr),
            )
        })?;
        Ok(PerfMetrics::from_whisper_decoded_results(ptr))
    }

    /// Construct from a raw pointer.
    pub(crate) fn from_ptr(ptr: *mut ov_genai_whisper_decoded_results) -> Self {
        Self { ptr }
    }
}

/// A timestamped chunk from Whisper speech recognition results.
pub struct WhisperDecodedResultChunk {
    ptr: *mut ov_genai_whisper_decoded_result_chunk,
}
drop_using_function!(
    WhisperDecodedResultChunk,
    ov_genai_whisper_decoded_result_chunk_free
);

impl WhisperDecodedResultChunk {
    /// Get the start timestamp in seconds.
    pub fn get_start_ts(&self) -> Result<f32> {
        let mut ts: f32 = 0.0;
        try_unsafe!(ov_genai_whisper_decoded_result_chunk_get_start_ts(
            self.ptr, &mut ts
        ))?;
        Ok(ts)
    }

    /// Get the end timestamp in seconds.
    pub fn get_end_ts(&self) -> Result<f32> {
        let mut ts: f32 = 0.0;
        try_unsafe!(ov_genai_whisper_decoded_result_chunk_get_end_ts(
            self.ptr, &mut ts
        ))?;
        Ok(ts)
    }

    /// Get the text for this chunk.
    pub fn get_text(&self) -> Result<String> {
        unsafe {
            crate::util::get_c_string_two_call(|buf, size| {
                ov_genai_whisper_decoded_result_chunk_get_text(self.ptr, buf, size)
            })
        }
    }

    /// Construct from a raw pointer.
    pub(crate) fn from_ptr(ptr: *mut ov_genai_whisper_decoded_result_chunk) -> Self {
        Self { ptr }
    }
}
