//! Safe wrapper for [`ov_genai_llm_pipeline`].

use crate::error::LoadingError;
use crate::{
    cstr, drop_using_function, try_unsafe, util::Result, ChatHistory, DecodedResults,
    GenerationConfig, SetupError, Streamer,
};
use openvino_genai_sys::{
    self, ov_genai_llm_pipeline, ov_genai_llm_pipeline_create, ov_genai_llm_pipeline_finish_chat,
    ov_genai_llm_pipeline_free, ov_genai_llm_pipeline_generate,
    ov_genai_llm_pipeline_generate_with_history, ov_genai_llm_pipeline_get_generation_config,
    ov_genai_llm_pipeline_set_generation_config, ov_genai_llm_pipeline_start_chat,
};

/// A pipeline for generating text using large language models.
///
/// This is the primary entry point for LLM inference with OpenVINO GenAI.
pub struct LlmPipeline {
    ptr: *mut ov_genai_llm_pipeline,
}
drop_using_function!(LlmPipeline, ov_genai_llm_pipeline_free);

unsafe impl Send for LlmPipeline {}

impl LlmPipeline {
    /// Create a new LLM pipeline from a model directory and device name.
    ///
    /// The `models_path` should point to a directory containing the model files exported for
    /// OpenVINO GenAI. The `device` can be `"CPU"`, `"GPU"`, `"NPU"`, etc.
    pub fn new(models_path: &str, device: &str) -> std::result::Result<Self, SetupError> {
        Self::with_properties(models_path, device, &[])
    }

    /// Create a new LLM pipeline with device properties.
    ///
    /// Properties are key-value string pairs passed to the underlying OpenVINO runtime.
    /// Common properties for NPU include:
    /// - `("CACHE_DIR", "/path/to/cache")` — cache compiled model blobs for faster reload
    /// - `("MAX_PROMPT_LEN", "128")` — maximum prompt length for NPU static shapes
    /// - `("MIN_RESPONSE_LEN", "64")` — minimum response length for NPU static shapes
    pub fn with_properties(
        models_path: &str,
        device: &str,
        properties: &[(&str, &str)],
    ) -> std::result::Result<Self, SetupError> {
        openvino_genai_sys::library::load().map_err(LoadingError::SystemFailure)?;
        let models_path = cstr!(models_path);
        let device = cstr!(device);
        let prop_cstrings: Vec<_> = properties
            .iter()
            .flat_map(|(k, v)| [cstr!(*k), cstr!(*v)])
            .collect();
        let prop_ptrs: Vec<_> = prop_cstrings.iter().map(|s| s.as_ptr()).collect();
        let mut ptr = std::ptr::null_mut();
        try_unsafe!(ov_genai_llm_pipeline_create(
            models_path.as_ptr(),
            device.as_ptr(),
            prop_ptrs.len(),
            std::ptr::addr_of_mut!(ptr),
            &prop_ptrs
        ))?;
        Ok(Self { ptr })
    }

    /// Generate text from a prompt.
    ///
    /// Optionally pass a [`GenerationConfig`] and/or a [`Streamer`] callback.
    pub fn generate(
        &mut self,
        prompt: &str,
        config: Option<&GenerationConfig>,
        streamer: Option<&Streamer>,
    ) -> Result<DecodedResults> {
        let prompt = cstr!(prompt);
        let config_ptr = config.map_or(std::ptr::null(), GenerationConfig::as_ptr);
        let streamer_raw = streamer.map(Streamer::as_raw);
        let streamer_ptr = streamer_raw
            .as_ref()
            .map_or(std::ptr::null(), std::ptr::from_ref);
        let mut results_ptr = std::ptr::null_mut();
        try_unsafe!(ov_genai_llm_pipeline_generate(
            self.ptr,
            prompt.as_ptr(),
            config_ptr,
            streamer_ptr,
            std::ptr::addr_of_mut!(results_ptr)
        ))?;
        Ok(DecodedResults::from_ptr(results_ptr))
    }

    /// Generate text using a chat history.
    ///
    /// Optionally pass a [`GenerationConfig`] and/or a [`Streamer`] callback.
    pub fn generate_with_history(
        &mut self,
        history: &ChatHistory,
        config: Option<&GenerationConfig>,
        streamer: Option<&Streamer>,
    ) -> Result<DecodedResults> {
        let config_ptr = config.map_or(std::ptr::null(), GenerationConfig::as_ptr);
        let streamer_raw = streamer.map(Streamer::as_raw);
        let streamer_ptr = streamer_raw
            .as_ref()
            .map_or(std::ptr::null(), std::ptr::from_ref);
        let mut results_ptr = std::ptr::null_mut();
        try_unsafe!(ov_genai_llm_pipeline_generate_with_history(
            self.ptr,
            history.as_ptr(),
            config_ptr,
            streamer_ptr,
            std::ptr::addr_of_mut!(results_ptr)
        ))?;
        Ok(DecodedResults::from_ptr(results_ptr))
    }

    /// Start a chat session, keeping history in the KV cache.
    pub fn start_chat(&mut self) -> Result<()> {
        try_unsafe!(ov_genai_llm_pipeline_start_chat(self.ptr))
    }

    /// Finish the chat session and clear the KV cache.
    pub fn finish_chat(&mut self) -> Result<()> {
        try_unsafe!(ov_genai_llm_pipeline_finish_chat(self.ptr))
    }

    /// Get a copy of the pipeline's current generation config.
    pub fn get_generation_config(&self) -> Result<GenerationConfig> {
        let mut ptr = std::ptr::null_mut();
        try_unsafe!(ov_genai_llm_pipeline_get_generation_config(
            self.ptr,
            std::ptr::addr_of_mut!(ptr)
        ))?;
        Ok(GenerationConfig::from_ptr(ptr))
    }

    /// Set the generation config for this pipeline.
    pub fn set_generation_config(&mut self, config: &mut GenerationConfig) -> Result<()> {
        try_unsafe!(ov_genai_llm_pipeline_set_generation_config(
            self.ptr, config.ptr
        ))
    }
}
