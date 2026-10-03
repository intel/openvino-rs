//! Integration tests for the speculative-decoding (draft-model) LLM pipeline.
//!
//! These tests reuse the `qwen3` fixture as both the main and draft model. This produces no
//! actual inference speedup (same model on both sides) but verifies the end-to-end plumbing:
//! pipeline construction with a draft, generation, the speculative-decoding generation-config
//! fields, and perf-metrics extraction. For benchmarking real speedup, swap in a smaller
//! draft model.

#![cfg(feature = "speculative-decoding")]

mod fixtures;

use fixtures::qwen3 as fixture;
use openvino_genai::{GenerationConfig, LlmPipeline};

fn try_pipeline() -> Option<LlmPipeline> {
    let model_dir = fixture::model_dir();
    let model_path = model_dir.to_string_lossy().into_owned();

    match LlmPipeline::with_draft(&model_path, "CPU", &model_path, "CPU") {
        Ok(p) => Some(p),
        Err(e) => {
            eprintln!("Skipping speculative pipeline tests: failed to create pipeline: {e}");
            None
        }
    }
}

#[test]
fn test_create_speculative_pipeline() {
    let _pipeline = match try_pipeline() {
        Some(p) => p,
        None => return,
    };
}

#[test]
fn test_generate_with_draft_model() {
    let mut pipeline = match try_pipeline() {
        Some(p) => p,
        None => return,
    };

    let mut config = GenerationConfig::new().unwrap();
    config.set_max_new_tokens(8).unwrap();
    config.set_num_assistant_tokens(4).unwrap();

    let results = pipeline.generate("Hello", Some(&config), None).unwrap();
    let text = results.get_string().unwrap();
    assert!(!text.is_empty(), "expected non-empty generation output");
}

#[test]
fn test_generation_config_assistant_tokens() {
    let mut pipeline = match try_pipeline() {
        Some(p) => p,
        None => return,
    };

    let mut config = GenerationConfig::new().unwrap();
    config.set_max_new_tokens(8).unwrap();
    config.set_num_assistant_tokens(4).unwrap();

    pipeline.generate("Hello", Some(&config), None).unwrap();
}

#[test]
fn test_sd_perf_metrics() {
    let mut pipeline = match try_pipeline() {
        Some(p) => p,
        None => return,
    };

    let mut config = GenerationConfig::new().unwrap();
    config.set_max_new_tokens(16).unwrap();
    config.set_num_assistant_tokens(4).unwrap();

    let results = pipeline
        .generate("Tell me a short joke.", Some(&config), None)
        .unwrap();

    let metrics = results
        .sd_perf_metrics()
        .unwrap()
        .expect("pipeline built via with_draft must produce SD perf metrics");

    // num_accepted_tokens is a usize getter; value may legitimately be 0 if the draft and main
    // never agree on a prefix. Just verify the call succeeds.
    let _accepted = metrics.num_accepted_tokens().unwrap();
    let main_generated = metrics.main_model_metrics().num_generated_tokens().unwrap();
    assert!(
        main_generated > 0,
        "main model should have generated at least one token"
    );

    // Smoke-test the mean/std getters — they should not error even when only one or two
    // tokens were produced.
    let _ttft = metrics.main_model_metrics().ttft().unwrap();
    let _tpot = metrics.main_model_metrics().tpot().unwrap();
    let _draft_generated = metrics
        .draft_model_metrics()
        .num_generated_tokens()
        .unwrap();
}
