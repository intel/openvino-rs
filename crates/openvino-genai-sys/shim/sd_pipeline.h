// Copyright (C) 2026 openvino-rs contributors
// SPDX-License-Identifier: Apache-2.0
//
// Flat C ABI exposing draft-model speculative decoding from OpenVINO GenAI's
// C++ API. The public OpenVINO GenAI C API (runtime/include/openvino/genai/c)
// does not expose a draft-model entry point, so this shim wraps the C++
// `ov::genai::LLMPipeline(main, device, draft_model(...))` constructor and
// the `SDPerModelsPerfMetrics` payload behind `DecodedResults::extended_perf_metrics`.
//
// The pipeline created by this shim is returned as a public
// `ov_genai_llm_pipeline*`. The shim reproduces the public C-API's opaque
// struct layout internally (see `types_c.h` in the openvino.genai source
// tree); the returned handle is therefore interchangeable with the public
// C-API and can be passed to `ov_genai_llm_pipeline_generate`,
// `ov_genai_llm_pipeline_free`, etc.

#pragma once

#include <stddef.h>

#include "openvino/c/ov_common.h"           // ov_status_e
#include "openvino/genai/c/llm_pipeline.h"  // ov_genai_llm_pipeline, ov_genai_decoded_results

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ov_genai_sd_perf_metrics_t ov_genai_sd_perf_metrics;

// Construct an LLMPipeline with a draft (assistant) model for speculative
// decoding. The resulting handle is a fully-functional public
// `ov_genai_llm_pipeline*` — use the public C-API for generate, chat, etc.,
// and free with `ov_genai_llm_pipeline_free`.
//
// Properties are flat key/value pairs `[k0, v0, k1, v1, ...]`, length
// `2 * n_props`. All keys and values are NUL-terminated UTF-8. Either device
// may be `NULL` (treated as empty string — OpenVINO picks the default).
ov_status_e ov_genai_sd_create_with_draft(
    const char* main_path,
    const char* main_device,
    size_t n_main_props,    const char* const* main_kv_flat,
    const char* draft_path,
    const char* draft_device,
    size_t n_draft_props,   const char* const* draft_kv_flat,
    ov_genai_llm_pipeline** out);

// Extract the speculative-decoding perf metrics from a `DecodedResults`. If
// the result's `extended_perf_metrics` is not an `SDPerModelsPerfMetrics`,
// `*out` is set to `NULL` and `OK` is returned.
ov_status_e ov_genai_sd_get_perf_metrics(
    const ov_genai_decoded_results* results,
    ov_genai_sd_perf_metrics** out);

void ov_genai_sd_perf_metrics_free(ov_genai_sd_perf_metrics* metrics);

ov_status_e ov_genai_sd_perf_metrics_get_num_accepted_tokens(
    const ov_genai_sd_perf_metrics* metrics,
    size_t* out);

// `side`: 0 = main model, 1 = draft model.
ov_status_e ov_genai_sd_perf_metrics_get_ttft     (const ov_genai_sd_perf_metrics* m, int side, float* mean, float* std);
ov_status_e ov_genai_sd_perf_metrics_get_ttst     (const ov_genai_sd_perf_metrics* m, int side, float* mean, float* std);
ov_status_e ov_genai_sd_perf_metrics_get_tpot     (const ov_genai_sd_perf_metrics* m, int side, float* mean, float* std);
ov_status_e ov_genai_sd_perf_metrics_get_latency  (const ov_genai_sd_perf_metrics* m, int side, float* mean, float* std);
ov_status_e ov_genai_sd_perf_metrics_get_generate_duration(const ov_genai_sd_perf_metrics* m, int side, float* mean, float* std);
ov_status_e ov_genai_sd_perf_metrics_get_num_generated_tokens(const ov_genai_sd_perf_metrics* m, int side, size_t* out);

#ifdef __cplusplus
}
#endif
