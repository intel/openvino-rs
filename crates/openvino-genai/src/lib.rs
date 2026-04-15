//! The [openvino-genai] crate provides high-level, ergonomic, safe Rust bindings to OpenVINO GenAI.
//! See the repository [README] for more information, such as build instructions.
//!
//! [openvino-genai]: https://crates.io/crates/openvino-genai
//! [README]: https://github.com/intel/openvino-rs
//!
//! Most interaction with OpenVINO GenAI begins with instantiating an [`LlmPipeline`]:
//! ```no_run
//! let pipeline = openvino_genai::LlmPipeline::new("path/to/model", "CPU")
//!     .expect("to create an LLM pipeline");
//! ```

#![deny(missing_docs)]
#![deny(clippy::all)]
#![warn(clippy::pedantic)]
#![warn(clippy::cargo)]
#![allow(
    clippy::must_use_candidate,
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::len_without_is_empty,
    clippy::doc_markdown
)]

mod chat_history;
mod chat_message;
mod decoded_results;
mod error;
mod generation_config;
mod json_container;
mod llm_pipeline;
mod perf_metrics;
mod streamer;
mod util;
mod vlm_pipeline;
mod whisper_generation_config;
mod whisper_pipeline;

pub use error::{InferenceError, LoadingError, SetupError};

/// Load the OpenVINO GenAI shared library using automatic discovery.
///
/// If the library has already been loaded, this is a no-op.
/// Delegates to [`openvino_genai_sys::library::load`].
pub fn load() -> Result<(), String> {
    openvino_genai_sys::library::load()
}

/// Load the OpenVINO GenAI shared library from an explicit path.
///
/// The `path` should point to the `openvino_genai_c` shared library file
/// (e.g., `libopenvino_genai_c.so`).
/// Delegates to [`openvino_genai_sys::library::load_from`].
pub fn load_from(path: impl Into<std::path::PathBuf>) -> Result<(), String> {
    openvino_genai_sys::library::load_from(path)
}
pub use chat_history::ChatHistory;
pub use chat_message::{ChatMessage, ToolCall};
pub use decoded_results::DecodedResults;
pub use generation_config::GenerationConfig;
pub use json_container::JsonContainer;
pub use llm_pipeline::LlmPipeline;
pub use perf_metrics::PerfMetrics;
pub use streamer::{Streamer, StreamingStatus};
pub use vlm_pipeline::{VlmDecodedResults, VlmPipeline};
pub use whisper_generation_config::WhisperGenerationConfig;
pub use whisper_pipeline::{WhisperDecodedResultChunk, WhisperDecodedResults, WhisperPipeline};
