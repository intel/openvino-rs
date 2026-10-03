// Copyright (C) 2026 openvino-rs contributors
// SPDX-License-Identifier: Apache-2.0
//
// Implementation of the speculative-decoding shim. See sd_pipeline.h.

#include "sd_pipeline.h"

#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/perf_metrics.hpp"
#include "openvino/genai/speculative_decoding/perf_metrics.hpp"

#include <cstring>
#include <exception>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

// Public-opaque layouts ------------------------------------------------------
//
// The public C-API headers forward-declare these as opaque types. Their
// internal layout is defined in `src/c/src/types_c.h` of the openvino.genai
// source tree as `struct { std::shared_ptr<CppType> object; }`. We reproduce
// the relevant layouts here so this shim can allocate a public opaque (the
// pipeline) and unwrap one (the decoded results) without changes to OpenVINO.
//
// If a future OpenVINO version changes these layouts, this shim will break;
// the layouts have been stable across all OpenVINO 2024.x / 2025.x / 2026.x
// releases to date.

struct ov_genai_llm_pipeline_opaque {
    std::shared_ptr<ov::genai::LLMPipeline> object;
};

struct ov_genai_decoded_results_opaque {
    std::shared_ptr<ov::genai::DecodedResults> object;
};

struct ov_genai_sd_perf_metrics_t {
    std::shared_ptr<ov::genai::SDPerModelsPerfMetrics> metrics;
};

// Helpers --------------------------------------------------------------------

namespace {

template <typename F>
ov_status_e guarded(F&& fn) {
    try {
        fn();
        return OK;
    } catch (const std::exception&) {
        return GENERAL_ERROR;
    } catch (...) {
        return UNKNOW_EXCEPTION;
    }
}

ov::AnyMap make_anymap(size_t n_pairs, const char* const* kv_flat) {
    ov::AnyMap m;
    if (kv_flat == nullptr) {
        return m;
    }
    for (size_t i = 0; i < n_pairs; ++i) {
        const char* k = kv_flat[2 * i];
        const char* v = kv_flat[2 * i + 1];
        if (k == nullptr || v == nullptr) {
            continue;
        }
        m[std::string(k)] = std::string(v);
    }
    return m;
}

std::string c_string(const char* s) {
    return s ? std::string(s) : std::string();
}

const ov::genai::SDPerfMetrics* side_metrics(const ov_genai_sd_perf_metrics* m, int side) {
    if (!m || !m->metrics) {
        return nullptr;
    }
    if (side == 0) {
        return &m->metrics->main_model_metrics;
    }
    if (side == 1) {
        return &m->metrics->draft_model_metrics;
    }
    return nullptr;
}

ov_status_e write_mean_std(ov::genai::MeanStdPair pair, float* mean, float* std) {
    if (mean) {
        *mean = pair.mean;
    }
    if (std) {
        *std = pair.std;
    }
    return OK;
}

}  // namespace

// Pipeline -------------------------------------------------------------------

extern "C" ov_status_e ov_genai_sd_create_with_draft(
    const char* main_path,
    const char* main_device,
    size_t n_main_props,    const char* const* main_kv_flat,
    const char* draft_path,
    const char* draft_device,
    size_t n_draft_props,   const char* const* draft_kv_flat,
    ov_genai_llm_pipeline** out)
{
    if (!main_path || !draft_path || !out) {
        return INVALID_C_PARAM;
    }
    return guarded([&] {
        ov::AnyMap main_props  = make_anymap(n_main_props,  main_kv_flat);
        ov::AnyMap draft_props = make_anymap(n_draft_props, draft_kv_flat);

        auto draft_entry = ov::genai::draft_model(
            std::filesystem::path(draft_path),
            c_string(draft_device),
            draft_props);
        main_props.insert(draft_entry);

        auto handle = std::make_unique<ov_genai_llm_pipeline_opaque>();
        handle->object = std::make_shared<ov::genai::LLMPipeline>(
            std::filesystem::path(main_path),
            c_string(main_device),
            main_props);
        *out = handle.release();
    });
}

// Perf metrics ---------------------------------------------------------------

extern "C" ov_status_e ov_genai_sd_get_perf_metrics(
    const ov_genai_decoded_results* results,
    ov_genai_sd_perf_metrics** out)
{
    if (!results || !out) {
        return INVALID_C_PARAM;
    }
    auto* opaque = reinterpret_cast<const ov_genai_decoded_results_opaque*>(results);
    if (!opaque->object) {
        return INVALID_C_PARAM;
    }
    return guarded([&] {
        auto ep = opaque->object->extended_perf_metrics;
        auto sd = std::dynamic_pointer_cast<ov::genai::SDPerModelsPerfMetrics>(ep);
        if (!sd) {
            *out = nullptr;
            return;
        }
        *out = new ov_genai_sd_perf_metrics_t{std::move(sd)};
    });
}

extern "C" void ov_genai_sd_perf_metrics_free(ov_genai_sd_perf_metrics* metrics) {
    delete metrics;
}

extern "C" ov_status_e ov_genai_sd_perf_metrics_get_num_accepted_tokens(
    const ov_genai_sd_perf_metrics* metrics,
    size_t* out)
{
    if (!metrics || !metrics->metrics || !out) {
        return INVALID_C_PARAM;
    }
    return guarded([&] { *out = metrics->metrics->get_num_accepted_tokens(); });
}

#define DEFINE_SIDE_MEAN_STD(suffix, expr)                                                  \
    extern "C" ov_status_e ov_genai_sd_perf_metrics_get_##suffix(                           \
        const ov_genai_sd_perf_metrics* m, int side, float* mean, float* std) {             \
        auto* side_m = side_metrics(m, side);                                               \
        if (!side_m) return INVALID_C_PARAM;                                                \
        return guarded([&] {                                                                \
            auto& mutable_m = const_cast<ov::genai::SDPerfMetrics&>(*side_m);               \
            write_mean_std(expr, mean, std);                                                \
        });                                                                                  \
    }

DEFINE_SIDE_MEAN_STD(ttft,               mutable_m.get_ttft())
DEFINE_SIDE_MEAN_STD(ttst,               mutable_m.get_ttst())
DEFINE_SIDE_MEAN_STD(tpot,               mutable_m.get_tpot())
DEFINE_SIDE_MEAN_STD(latency,            mutable_m.get_latency())
DEFINE_SIDE_MEAN_STD(generate_duration,  mutable_m.get_generate_duration())

#undef DEFINE_SIDE_MEAN_STD

extern "C" ov_status_e ov_genai_sd_perf_metrics_get_num_generated_tokens(
    const ov_genai_sd_perf_metrics* m, int side, size_t* out)
{
    auto* side_m = side_metrics(m, side);
    if (!side_m || !out) {
        return INVALID_C_PARAM;
    }
    return guarded([&] {
        auto& mutable_m = const_cast<ov::genai::SDPerfMetrics&>(*side_m);
        *out = mutable_m.get_num_generated_tokens();
    });
}
