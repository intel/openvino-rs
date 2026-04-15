//! Safe wrapper for [`ov_genai_whisper_generation_config`].

use crate::{cstr, drop_using_function, try_unsafe, util::Result, GenerationConfig};
use openvino_genai_sys::{
    self, ov_genai_whisper_generation_config, ov_genai_whisper_generation_config_create,
    ov_genai_whisper_generation_config_create_from_json, ov_genai_whisper_generation_config_free,
    ov_genai_whisper_generation_config_get_generation_config,
    ov_genai_whisper_generation_config_get_is_multilingual,
    ov_genai_whisper_generation_config_get_return_timestamps,
    ov_genai_whisper_generation_config_set_is_multilingual,
    ov_genai_whisper_generation_config_set_language,
    ov_genai_whisper_generation_config_set_return_timestamps,
    ov_genai_whisper_generation_config_set_task, ov_genai_whisper_generation_config_validate,
};

/// Configuration for Whisper speech recognition.
pub struct WhisperGenerationConfig {
    ptr: *mut ov_genai_whisper_generation_config,
}
drop_using_function!(
    WhisperGenerationConfig,
    ov_genai_whisper_generation_config_free
);

impl WhisperGenerationConfig {
    /// Create a new default whisper generation config.
    pub fn new() -> Result<Self> {
        let mut ptr = std::ptr::null_mut();
        try_unsafe!(ov_genai_whisper_generation_config_create(
            std::ptr::addr_of_mut!(ptr)
        ))?;
        Ok(Self { ptr })
    }

    /// Create a whisper generation config from a JSON file.
    pub fn from_json(json_path: &str) -> Result<Self> {
        let json_path = cstr!(json_path);
        let mut ptr = std::ptr::null_mut();
        try_unsafe!(ov_genai_whisper_generation_config_create_from_json(
            json_path.as_ptr(),
            std::ptr::addr_of_mut!(ptr)
        ))?;
        Ok(Self { ptr })
    }

    /// Get the underlying generation config.
    pub fn get_generation_config(&self) -> Result<GenerationConfig> {
        let mut ptr = std::ptr::null_mut();
        try_unsafe!(ov_genai_whisper_generation_config_get_generation_config(
            self.ptr,
            std::ptr::addr_of_mut!(ptr)
        ))?;
        Ok(GenerationConfig::from_ptr(ptr))
    }

    /// Set the language for generation (e.g., `"en"`, `"fr"`, `"de"`).
    pub fn set_language(&mut self, language: &str) -> Result<()> {
        let language = cstr!(language);
        try_unsafe!(ov_genai_whisper_generation_config_set_language(
            self.ptr,
            language.as_ptr()
        ))
    }

    /// Get the language for generation.
    pub fn get_language(&self) -> Result<String> {
        unsafe {
            crate::util::get_c_string_two_call(|buf, size| {
                openvino_genai_sys::ov_genai_whisper_generation_config_get_language(
                    self.ptr, buf, size,
                )
            })
        }
    }

    /// Set the task (`"translate"` or `"transcribe"`).
    pub fn set_task(&mut self, task: &str) -> Result<()> {
        let task = cstr!(task);
        try_unsafe!(ov_genai_whisper_generation_config_set_task(
            self.ptr,
            task.as_ptr()
        ))
    }

    /// Get the task for generation.
    pub fn get_task(&self) -> Result<String> {
        unsafe {
            crate::util::get_c_string_two_call(|buf, size| {
                openvino_genai_sys::ov_genai_whisper_generation_config_get_task(self.ptr, buf, size)
            })
        }
    }

    /// Set whether the model is multilingual.
    pub fn set_is_multilingual(&mut self, value: bool) -> Result<()> {
        try_unsafe!(ov_genai_whisper_generation_config_set_is_multilingual(
            self.ptr, value
        ))
    }

    /// Get whether the model is multilingual.
    pub fn get_is_multilingual(&self) -> Result<bool> {
        let mut value = false;
        try_unsafe!(ov_genai_whisper_generation_config_get_is_multilingual(
            self.ptr, &mut value
        ))?;
        Ok(value)
    }

    /// Set whether to return timestamps with segments.
    pub fn set_return_timestamps(&mut self, value: bool) -> Result<()> {
        try_unsafe!(ov_genai_whisper_generation_config_set_return_timestamps(
            self.ptr, value
        ))
    }

    /// Get whether timestamps will be returned.
    pub fn get_return_timestamps(&self) -> Result<bool> {
        let mut value = false;
        try_unsafe!(ov_genai_whisper_generation_config_get_return_timestamps(
            self.ptr, &mut value
        ))?;
        Ok(value)
    }

    /// Validate the configuration.
    pub fn validate(&mut self) -> Result<()> {
        try_unsafe!(ov_genai_whisper_generation_config_validate(self.ptr))
    }

    /// Construct from a raw pointer.
    pub(crate) fn from_ptr(ptr: *mut ov_genai_whisper_generation_config) -> Self {
        Self { ptr }
    }

    /// Get the raw pointer.
    pub(crate) fn as_ptr(&self) -> *const ov_genai_whisper_generation_config {
        self.ptr
    }

    /// Get a mutable raw pointer.
    pub(crate) fn as_mut_ptr(&mut self) -> *mut ov_genai_whisper_generation_config {
        self.ptr
    }
}
