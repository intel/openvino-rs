//! Hand-written FFI for the speculative-decoding shim compiled in `shim/sd_pipeline.cpp`.
//!
//! The shim exposes draft-model speculative decoding through a small C ABI built on top of the
//! public C-API's opaque types — the pipeline handle returned by
//! [`ov_genai_sd_create_with_draft`] is interchangeable with the public `ov_genai_llm_pipeline`
//! and must be freed by [`ov_genai_llm_pipeline_free`](crate::ov_genai_llm_pipeline_free).
//!
//! The shim depends on `libopenvino_genai` (the C++ library), so this feature requires
//! `dynamic-linking` and is incompatible with `runtime-linking`.

#![cfg(feature = "speculative-decoding")]
#![allow(non_camel_case_types)]

#[cfg(feature = "runtime-linking")]
compile_error!(
    "the `speculative-decoding` feature requires `dynamic-linking`; \
     it is incompatible with `runtime-linking`"
);

use crate::{ov_genai_decoded_results, ov_genai_llm_pipeline, ov_status_e};

/// Opaque handle to a wrapped `ov::genai::SDPerModelsPerfMetrics`.
#[repr(C)]
pub struct ov_genai_sd_perf_metrics {
    _unused: [u8; 0],
}

unsafe extern "C" {
    /// Create an LLMPipeline with a draft (assistant) model. The returned handle is
    /// interchangeable with the public C-API pipeline opaque and must be freed by
    /// `ov_genai_llm_pipeline_free`.
    pub fn ov_genai_sd_create_with_draft(
        main_path: *const ::std::os::raw::c_char,
        main_device: *const ::std::os::raw::c_char,
        n_main_props: usize,
        main_kv_flat: *const *const ::std::os::raw::c_char,
        draft_path: *const ::std::os::raw::c_char,
        draft_device: *const ::std::os::raw::c_char,
        n_draft_props: usize,
        draft_kv_flat: *const *const ::std::os::raw::c_char,
        out: *mut *mut ov_genai_llm_pipeline,
    ) -> ov_status_e;

    /// Extract speculative-decoding perf metrics from a `DecodedResults`. Sets `*out = NULL`
    /// and returns `OK` if the result's `extended_perf_metrics` is not an
    /// `SDPerModelsPerfMetrics`.
    pub fn ov_genai_sd_get_perf_metrics(
        results: *const ov_genai_decoded_results,
        out: *mut *mut ov_genai_sd_perf_metrics,
    ) -> ov_status_e;

    pub fn ov_genai_sd_perf_metrics_free(metrics: *mut ov_genai_sd_perf_metrics);

    pub fn ov_genai_sd_perf_metrics_get_num_accepted_tokens(
        metrics: *const ov_genai_sd_perf_metrics,
        out: *mut usize,
    ) -> ov_status_e;

    pub fn ov_genai_sd_perf_metrics_get_ttft(
        m: *const ov_genai_sd_perf_metrics,
        side: ::std::os::raw::c_int,
        mean: *mut f32,
        std: *mut f32,
    ) -> ov_status_e;
    pub fn ov_genai_sd_perf_metrics_get_ttst(
        m: *const ov_genai_sd_perf_metrics,
        side: ::std::os::raw::c_int,
        mean: *mut f32,
        std: *mut f32,
    ) -> ov_status_e;
    pub fn ov_genai_sd_perf_metrics_get_tpot(
        m: *const ov_genai_sd_perf_metrics,
        side: ::std::os::raw::c_int,
        mean: *mut f32,
        std: *mut f32,
    ) -> ov_status_e;
    pub fn ov_genai_sd_perf_metrics_get_latency(
        m: *const ov_genai_sd_perf_metrics,
        side: ::std::os::raw::c_int,
        mean: *mut f32,
        std: *mut f32,
    ) -> ov_status_e;
    pub fn ov_genai_sd_perf_metrics_get_generate_duration(
        m: *const ov_genai_sd_perf_metrics,
        side: ::std::os::raw::c_int,
        mean: *mut f32,
        std: *mut f32,
    ) -> ov_status_e;
    pub fn ov_genai_sd_perf_metrics_get_num_generated_tokens(
        m: *const ov_genai_sd_perf_metrics,
        side: ::std::os::raw::c_int,
        out: *mut usize,
    ) -> ov_status_e;
}
