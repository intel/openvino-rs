/* manually created from OpenVINO GenAI C API headers */

// Re-use the ov_status_e enum from the OpenVINO C API. The GenAI C API returns the same status
// codes.
#[repr(i32)]
#[doc = " @enum ov_status_e\n @ingroup ov_base_c_api\n @brief This enum contains codes for all possible return values of the interface functions"]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum ov_status_e {
    #[doc = "!< SUCCESS"]
    OK = 0,
    #[doc = "!< GENERAL_ERROR"]
    GENERAL_ERROR = -1,
    #[doc = "!< NOT_IMPLEMENTED"]
    NOT_IMPLEMENTED = -2,
    #[doc = "!< NETWORK_NOT_LOADED"]
    NETWORK_NOT_LOADED = -3,
    #[doc = "!< PARAMETER_MISMATCH"]
    PARAMETER_MISMATCH = -4,
    #[doc = "!< NOT_FOUND"]
    NOT_FOUND = -5,
    #[doc = "!< OUT_OF_BOUNDS"]
    OUT_OF_BOUNDS = -6,
    #[doc = "!< UNEXPECTED"]
    UNEXPECTED = -7,
    #[doc = "!< REQUEST_BUSY"]
    REQUEST_BUSY = -8,
    #[doc = "!< RESULT_NOT_READY"]
    RESULT_NOT_READY = -9,
    #[doc = "!< NOT_ALLOCATED"]
    NOT_ALLOCATED = -10,
    #[doc = "!< INFER_NOT_STARTED"]
    INFER_NOT_STARTED = -11,
    #[doc = "!< NETWORK_NOT_READ"]
    NETWORK_NOT_READ = -12,
    #[doc = "!< INFER_CANCELLED"]
    INFER_CANCELLED = -13,
    #[doc = "!< INVALID_C_PARAM"]
    INVALID_C_PARAM = -14,
    #[doc = "!< UNKNOWN_C_ERROR"]
    UNKNOWN_C_ERROR = -15,
    #[doc = "!< NOT_IMPLEMENT_C_METHOD"]
    NOT_IMPLEMENT_C_METHOD = -16,
    #[doc = "!< UNKNOW_EXCEPTION"]
    UNKNOW_EXCEPTION = -17,
}

// --- Streaming status ---

#[repr(i32)]
#[doc = " Streaming status for streamer callbacks."]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum ov_genai_streaming_status_e {
    #[doc = "Continue to run inference"]
    RUNNING = 0,
    #[doc = "Stop generation, keep history as is"]
    STOP = 1,
    #[doc = "Stop generation, drop last prompt and generated tokens from history"]
    CANCEL = 2,
}

// --- Streamer callback ---

#[doc = " Structure for streamer callback functions with arguments."]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct streamer_callback {
    #[doc = "Pointer to the callback function"]
    pub callback_func:
        Option<unsafe extern "C" fn(str_: *const ::std::os::raw::c_char, args: *mut ::std::os::raw::c_void) -> ov_genai_streaming_status_e>,
    #[doc = "Pointer to the arguments passed to the callback function"]
    pub args: *mut ::std::os::raw::c_void,
}

// --- StopCriteria enum ---

#[repr(i32)]
#[doc = " Controls the stopping condition for grouped beam search."]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum StopCriteria {
    #[doc = "Stops as soon as there are `num_beams` complete candidates."]
    EARLY = 0,
    #[doc = "Stops when it is unlikely to find better candidates."]
    HEURISTIC = 1,
    #[doc = "Stops when there cannot be better candidates."]
    NEVER = 2,
}

// --- Opaque types ---

#[doc = " Opaque type for LLMPipeline."]
#[repr(C)]
pub struct ov_genai_llm_pipeline {
    _private: [u8; 0],
}

#[doc = " Opaque type for GenerationConfig."]
#[repr(C)]
pub struct ov_genai_generation_config {
    _private: [u8; 0],
}

#[doc = " Opaque type for decoded results from LLM generation."]
#[repr(C)]
pub struct ov_genai_decoded_results {
    _private: [u8; 0],
}

#[doc = " Opaque type for ChatHistory."]
#[repr(C)]
pub struct ov_genai_chat_history {
    _private: [u8; 0],
}

#[doc = " Opaque type for JsonContainer."]
#[repr(C)]
pub struct ov_genai_json_container {
    _private: [u8; 0],
}

#[doc = " Opaque type for VLMPipeline."]
#[repr(C)]
pub struct ov_genai_vlm_pipeline {
    _private: [u8; 0],
}

#[doc = " Opaque type for decoded results from VLM generation."]
#[repr(C)]
pub struct ov_genai_vlm_decoded_results {
    _private: [u8; 0],
}

#[doc = " Opaque type for WhisperPipeline."]
#[repr(C)]
pub struct ov_genai_whisper_pipeline {
    _private: [u8; 0],
}

#[doc = " Opaque type for WhisperGenerationConfig."]
#[repr(C)]
pub struct ov_genai_whisper_generation_config {
    _private: [u8; 0],
}

#[doc = " Opaque type for decoded results from Whisper generation."]
#[repr(C)]
pub struct ov_genai_whisper_decoded_results {
    _private: [u8; 0],
}

#[doc = " Opaque type for a chunk from Whisper decoded results."]
#[repr(C)]
pub struct ov_genai_whisper_decoded_result_chunk {
    _private: [u8; 0],
}

#[doc = " Opaque type for performance metrics."]
#[repr(C)]
pub struct ov_genai_perf_metrics {
    _private: [u8; 0],
}

// --- Opaque type from core OpenVINO used by VLM pipeline for image tensors ---

#[doc = " Opaque type for ov_tensor_t (from core OpenVINO C API)."]
#[repr(C)]
pub struct ov_tensor_t {
    _private: [u8; 0],
}

// --- ChatHistory status codes ---

#[repr(i32)]
#[doc = " Status codes for chat history operations."]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum ov_genai_chat_history_status_e {
    OK = 0,
    INVALID_PARAM = -1,
    OUT_OF_BOUNDS = -2,
    EMPTY = -3,
    INVALID_JSON = -4,
    ERROR = -5,
}

// --- JsonContainer status codes ---

#[repr(i32)]
#[doc = " Status codes for JsonContainer operations."]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum ov_genai_json_container_status_e {
    OK = 0,
    INVALID_PARAM = -1,
    INVALID_JSON = -2,
    OUT_OF_BOUNDS = -3,
    ERROR = -4,
}
