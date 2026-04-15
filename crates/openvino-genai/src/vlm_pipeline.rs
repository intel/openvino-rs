//! Safe wrapper for [`ov_genai_vlm_pipeline`] — Vision-Language Model pipeline.

use crate::error::LoadingError;
use crate::{
    cstr, drop_using_function, try_unsafe, util::Result, GenerationConfig, PerfMetrics, SetupError,
    Streamer,
};
use openvino_genai_sys::{
    self, ov_genai_vlm_decoded_results, ov_genai_vlm_decoded_results_free,
    ov_genai_vlm_decoded_results_get_perf_metrics, ov_genai_vlm_decoded_results_get_string,
    ov_genai_vlm_pipeline, ov_genai_vlm_pipeline_create, ov_genai_vlm_pipeline_finish_chat,
    ov_genai_vlm_pipeline_free, ov_genai_vlm_pipeline_generate,
    ov_genai_vlm_pipeline_get_generation_config, ov_genai_vlm_pipeline_set_generation_config,
    ov_genai_vlm_pipeline_start_chat, ov_tensor_t,
};

/// A pipeline for generating text from text+image inputs using Vision-Language Models.
pub struct VlmPipeline {
    ptr: *mut ov_genai_vlm_pipeline,
}
drop_using_function!(VlmPipeline, ov_genai_vlm_pipeline_free);

unsafe impl Send for VlmPipeline {}

impl VlmPipeline {
    /// Create a new VLM pipeline from a model directory and device name.
    pub fn new(models_path: &str, device: &str) -> std::result::Result<Self, SetupError> {
        Self::with_properties(models_path, device, &[])
    }

    /// Create a new VLM pipeline with device properties.
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
        try_unsafe!(ov_genai_vlm_pipeline_create(
            models_path.as_ptr(),
            device.as_ptr(),
            prop_ptrs.len(),
            std::ptr::addr_of_mut!(ptr),
            &prop_ptrs
        ))?;
        Ok(Self { ptr })
    }

    /// Generate text from a text prompt and optional image tensors.
    ///
    /// The `images` parameter takes raw `ov_tensor_t` pointers for image data. Users with the
    /// `openvino` crate can obtain these from `openvino::Tensor`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the `ov_tensor_t` pointers in `images` are valid and live for
    /// the duration of this call.
    pub fn generate(
        &mut self,
        prompt: &str,
        images: &[*const ov_tensor_t],
        config: Option<&GenerationConfig>,
        streamer: Option<&Streamer>,
    ) -> Result<VlmDecodedResults> {
        let prompt = cstr!(prompt);
        let config_ptr = config.map_or(std::ptr::null(), GenerationConfig::as_ptr);
        let streamer_raw = streamer.map(Streamer::as_raw);
        let streamer_ptr = streamer_raw
            .as_ref()
            .map_or(std::ptr::null(), std::ptr::from_ref);
        let images_ptr = if images.is_empty() {
            std::ptr::null_mut()
        } else {
            images.as_ptr().cast_mut()
        };
        let mut results_ptr = std::ptr::null_mut();
        try_unsafe!(ov_genai_vlm_pipeline_generate(
            self.ptr,
            prompt.as_ptr(),
            images_ptr,
            images.len(),
            config_ptr,
            streamer_ptr,
            std::ptr::addr_of_mut!(results_ptr)
        ))?;
        Ok(VlmDecodedResults::from_ptr(results_ptr))
    }

    /// Start a chat session.
    pub fn start_chat(&mut self) -> Result<()> {
        try_unsafe!(ov_genai_vlm_pipeline_start_chat(self.ptr))
    }

    /// Finish the chat session.
    pub fn finish_chat(&mut self) -> Result<()> {
        try_unsafe!(ov_genai_vlm_pipeline_finish_chat(self.ptr))
    }

    /// Get the pipeline's current generation config.
    pub fn get_generation_config(&self) -> Result<GenerationConfig> {
        let mut ptr = std::ptr::null_mut();
        try_unsafe!(ov_genai_vlm_pipeline_get_generation_config(
            self.ptr,
            std::ptr::addr_of_mut!(ptr)
        ))?;
        Ok(GenerationConfig::from_ptr(ptr))
    }

    /// Set the generation config for this pipeline.
    pub fn set_generation_config(&mut self, config: &mut GenerationConfig) -> Result<()> {
        try_unsafe!(ov_genai_vlm_pipeline_set_generation_config(
            self.ptr, config.ptr
        ))
    }
}

/// Results from VLM text generation.
pub struct VlmDecodedResults {
    ptr: *mut ov_genai_vlm_decoded_results,
}
drop_using_function!(VlmDecodedResults, ov_genai_vlm_decoded_results_free);

impl VlmDecodedResults {
    /// Get the generated text as a string.
    pub fn get_string(&self) -> Result<String> {
        unsafe {
            crate::util::get_c_string_two_call(|buf, size| {
                ov_genai_vlm_decoded_results_get_string(self.ptr, buf, size)
            })
        }
    }

    /// Get performance metrics from these results.
    pub fn get_perf_metrics(&self) -> Result<PerfMetrics> {
        let mut ptr = std::ptr::null_mut();
        crate::InferenceError::convert(unsafe {
            ov_genai_vlm_decoded_results_get_perf_metrics(self.ptr, std::ptr::addr_of_mut!(ptr))
        })?;
        Ok(PerfMetrics::from_vlm_decoded_results(ptr))
    }

    /// Construct from a raw pointer.
    pub(crate) fn from_ptr(ptr: *mut ov_genai_vlm_decoded_results) -> Self {
        Self { ptr }
    }
}
