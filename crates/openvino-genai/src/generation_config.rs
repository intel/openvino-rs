//! Safe wrapper for [`ov_genai_generation_config`].

use crate::{cstr, drop_using_function, try_unsafe, util::Result};
use openvino_genai_sys::{
    self, ov_genai_generation_config, ov_genai_generation_config_create,
    ov_genai_generation_config_create_from_json, ov_genai_generation_config_free,
    ov_genai_generation_config_get_max_new_tokens, ov_genai_generation_config_set_do_sample,
    ov_genai_generation_config_set_frequency_penalty, ov_genai_generation_config_set_max_length,
    ov_genai_generation_config_set_max_new_tokens, ov_genai_generation_config_set_num_beams,
    ov_genai_generation_config_set_presence_penalty,
    ov_genai_generation_config_set_repetition_penalty, ov_genai_generation_config_set_rng_seed,
    ov_genai_generation_config_set_temperature, ov_genai_generation_config_set_top_k,
    ov_genai_generation_config_set_top_p, ov_genai_generation_config_validate,
};

/// Configuration for text generation.
///
/// See the [OpenVINO GenAI documentation](https://docs.openvino.ai/2024/api/genai_api.html) for
/// details on each parameter.
pub struct GenerationConfig {
    pub(crate) ptr: *mut ov_genai_generation_config,
}
drop_using_function!(GenerationConfig, ov_genai_generation_config_free);

impl GenerationConfig {
    /// Create a new default generation config.
    pub fn new() -> Result<Self> {
        let mut ptr = std::ptr::null_mut();
        try_unsafe!(ov_genai_generation_config_create(std::ptr::addr_of_mut!(
            ptr
        )))?;
        Ok(Self { ptr })
    }

    /// Create a generation config from a JSON file.
    pub fn from_json(json_path: &str) -> Result<Self> {
        let json_path = cstr!(json_path);
        let mut ptr = std::ptr::null_mut();
        try_unsafe!(ov_genai_generation_config_create_from_json(
            json_path.as_ptr(),
            std::ptr::addr_of_mut!(ptr)
        ))?;
        Ok(Self { ptr })
    }

    /// Set the maximum number of tokens to generate.
    pub fn set_max_new_tokens(&mut self, value: usize) -> Result<()> {
        try_unsafe!(ov_genai_generation_config_set_max_new_tokens(
            self.ptr, value
        ))
    }

    /// Set the maximum total length (prompt + generated).
    pub fn set_max_length(&mut self, value: usize) -> Result<()> {
        try_unsafe!(ov_genai_generation_config_set_max_length(self.ptr, value))
    }

    /// Set the temperature for random sampling.
    pub fn set_temperature(&mut self, value: f32) -> Result<()> {
        try_unsafe!(ov_genai_generation_config_set_temperature(self.ptr, value))
    }

    /// Set the top-p (nucleus sampling) threshold.
    pub fn set_top_p(&mut self, value: f32) -> Result<()> {
        try_unsafe!(ov_genai_generation_config_set_top_p(self.ptr, value))
    }

    /// Set the top-k filtering value.
    pub fn set_top_k(&mut self, value: usize) -> Result<()> {
        try_unsafe!(ov_genai_generation_config_set_top_k(self.ptr, value))
    }

    /// Set whether to use multinomial random sampling.
    pub fn set_do_sample(&mut self, value: bool) -> Result<()> {
        try_unsafe!(ov_genai_generation_config_set_do_sample(self.ptr, value))
    }

    /// Set the number of beams for beam search. 1 disables beam search.
    pub fn set_num_beams(&mut self, value: usize) -> Result<()> {
        try_unsafe!(ov_genai_generation_config_set_num_beams(self.ptr, value))
    }

    /// Set the repetition penalty. 1.0 means no penalty.
    pub fn set_repetition_penalty(&mut self, value: f32) -> Result<()> {
        try_unsafe!(ov_genai_generation_config_set_repetition_penalty(
            self.ptr, value
        ))
    }

    /// Set the presence penalty.
    pub fn set_presence_penalty(&mut self, value: f32) -> Result<()> {
        try_unsafe!(ov_genai_generation_config_set_presence_penalty(
            self.ptr, value
        ))
    }

    /// Set the frequency penalty.
    pub fn set_frequency_penalty(&mut self, value: f32) -> Result<()> {
        try_unsafe!(ov_genai_generation_config_set_frequency_penalty(
            self.ptr, value
        ))
    }

    /// Set the random number generator seed.
    pub fn set_rng_seed(&mut self, value: usize) -> Result<()> {
        try_unsafe!(ov_genai_generation_config_set_rng_seed(self.ptr, value))
    }

    /// Get the maximum number of tokens to generate.
    pub fn get_max_new_tokens(&self) -> Result<usize> {
        let mut value: usize = 0;
        try_unsafe!(ov_genai_generation_config_get_max_new_tokens(
            self.ptr, &mut value
        ))?;
        Ok(value)
    }

    /// Set stop strings that will cause generation to stop.
    pub fn set_stop_strings(&mut self, strings: &[&str]) -> Result<()> {
        let c_strings: Vec<std::ffi::CString> = strings.iter().map(|s| cstr!(*s)).collect();
        let mut ptrs: Vec<*const std::os::raw::c_char> =
            c_strings.iter().map(|s| s.as_ptr()).collect();
        try_unsafe!(
            openvino_genai_sys::ov_genai_generation_config_set_stop_strings(
                self.ptr,
                ptrs.as_mut_ptr(),
                ptrs.len()
            )
        )
    }

    /// Set whether stop strings should be included in the output.
    pub fn set_include_stop_str_in_output(&mut self, value: bool) -> Result<()> {
        try_unsafe!(
            openvino_genai_sys::ov_genai_generation_config_set_include_stop_str_in_output(
                self.ptr, value
            )
        )
    }

    /// Validate the configuration for conflicting parameters.
    pub fn validate(&mut self) -> Result<()> {
        try_unsafe!(ov_genai_generation_config_validate(self.ptr))
    }

    /// Construct from a raw pointer. For internal use.
    pub(crate) fn from_ptr(ptr: *mut ov_genai_generation_config) -> Self {
        Self { ptr }
    }

    /// Get the raw pointer. For internal use.
    pub(crate) fn as_ptr(&self) -> *const ov_genai_generation_config {
        self.ptr
    }
}
