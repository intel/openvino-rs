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

mod error;
mod util;
mod generation_config;
mod decoded_results;
mod llm_pipeline;
mod vlm_pipeline;
mod whisper_pipeline;
mod whisper_generation_config;
mod chat_history;
mod chat_message;
mod json_container;
mod streamer;
mod perf_metrics;

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
pub use generation_config::GenerationConfig;
pub use decoded_results::DecodedResults;
pub use llm_pipeline::LlmPipeline;
pub use vlm_pipeline::{VlmPipeline, VlmDecodedResults};
pub use whisper_pipeline::{WhisperPipeline, WhisperDecodedResults, WhisperDecodedResultChunk};
pub use whisper_generation_config::WhisperGenerationConfig;
pub use chat_history::ChatHistory;
pub use chat_message::{ChatMessage, ToolCall};
pub use json_container::JsonContainer;
pub use streamer::{Streamer, StreamingStatus};
pub use perf_metrics::PerfMetrics;
