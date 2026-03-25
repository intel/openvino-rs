//! On-demand fixture downloads for openvino-genai integration tests.
//!
//! This module lazily downloads model files from HuggingFace using `curl`, following the same
//! pattern as the `openvino` crate's fixture system. Files are cached locally so subsequent test
//! runs skip the download.

#![allow(dead_code)]

use std::path::PathBuf;
use std::process::Command;

/// Download a single file from a HuggingFace model repository.
///
/// Returns the local path to the downloaded file. Skips the download if the file already exists.
fn download_hf(repo: &str, file: &str, local_dir: &str) -> anyhow::Result<PathBuf> {
    let to = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(local_dir)
        .join(file);
    if to.exists() {
        println!("> skipping: {local_dir}/{file}");
        return Ok(to);
    }

    // Ensure parent directories exist (for nested paths, if any).
    if let Some(parent) = to.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let url = format!("https://huggingface.co/{repo}/resolve/main/{file}");
    let mut curl = Command::new("curl");
    curl.arg("--location")
        .arg("--fail")
        .arg(&url)
        .arg("--output")
        .arg(&to);
    println!("> downloading: {local_dir}/{file}");
    let result = curl.output()?;
    if !result.status.success() {
        // Clean up partial download.
        let _ = std::fs::remove_file(&to);
        panic!(
            "curl failed for {url}: {}\n{}",
            result.status,
            String::from_utf8_lossy(&result.stderr)
        );
    }

    Ok(to)
}

/// Download all files for a model and return the directory path.
fn download_model(repo: &str, local_dir: &str, files: &[&str]) -> PathBuf {
    for file in files {
        download_hf(repo, file, local_dir).unwrap();
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(local_dir)
}

/// Qwen3-0.6B-int4-ov — a small LLM for testing.
pub mod qwen3 {
    use super::*;
    use std::path::PathBuf;

    const REPO: &str = "OpenVINO/Qwen3-0.6B-int4-ov";
    const FILES: &[&str] = &[
        "config.json",
        "generation_config.json",
        "openvino_detokenizer.bin",
        "openvino_detokenizer.xml",
        "openvino_model.bin",
        "openvino_model.xml",
        "openvino_tokenizer.bin",
        "openvino_tokenizer.xml",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ];

    pub fn model_dir() -> PathBuf {
        download_model(REPO, "qwen3", FILES)
    }
}

/// whisper-tiny-int8-ov — a small Whisper model for testing.
pub mod whisper_tiny {
    use super::*;
    use std::path::PathBuf;

    const REPO: &str = "OpenVINO/whisper-tiny-int8-ov";
    const FILES: &[&str] = &[
        "config.json",
        "generation_config.json",
        "normalizer.json",
        "openvino_config.json",
        "openvino_decoder_model.bin",
        "openvino_decoder_model.xml",
        "openvino_detokenizer.bin",
        "openvino_detokenizer.xml",
        "openvino_encoder_model.bin",
        "openvino_encoder_model.xml",
        "openvino_tokenizer.bin",
        "openvino_tokenizer.xml",
        "preprocessor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ];

    pub fn model_dir() -> PathBuf {
        download_model(REPO, "whisper-tiny", FILES)
    }
}

/// InternVL2-1B-int8-ov — a small VLM for testing.
pub mod internvl2 {
    use super::*;
    use std::path::PathBuf;

    const REPO: &str = "OpenVINO/InternVL2-1B-int8-ov";
    const FILES: &[&str] = &[
        "config.json",
        "generation_config.json",
        "openvino_config.json",
        "openvino_detokenizer.bin",
        "openvino_detokenizer.xml",
        "openvino_language_model.bin",
        "openvino_language_model.xml",
        "openvino_text_embeddings_model.bin",
        "openvino_text_embeddings_model.xml",
        "openvino_tokenizer.bin",
        "openvino_tokenizer.xml",
        "openvino_vision_embeddings_model.bin",
        "openvino_vision_embeddings_model.xml",
        "preprocessor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ];

    pub fn model_dir() -> PathBuf {
        download_model(REPO, "internvl2", FILES)
    }
}
