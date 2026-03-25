use super::types::*;
use crate::link;
link! {

// =============================================================================
// DecodedResults
// =============================================================================

unsafe extern "C" {
    #[doc = " Create DecodedResults."]
    pub fn ov_genai_decoded_results_create(results: *mut *mut ov_genai_decoded_results) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Release the memory allocated by ov_genai_decoded_results."]
    pub fn ov_genai_decoded_results_free(results: *mut ov_genai_decoded_results);
}
unsafe extern "C" {
    #[doc = " Get performance metrics from ov_genai_decoded_results."]
    pub fn ov_genai_decoded_results_get_perf_metrics(results: *const ov_genai_decoded_results, metrics: *mut *mut ov_genai_perf_metrics) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Release the memory allocated by ov_genai_perf_metrics from decoded results."]
    pub fn ov_genai_decoded_results_perf_metrics_free(metrics: *mut ov_genai_perf_metrics);
}
unsafe extern "C" {
    #[doc = " Get string result from ov_genai_decoded_results."]
    pub fn ov_genai_decoded_results_get_string(results: *const ov_genai_decoded_results, output: *mut ::std::os::raw::c_char, output_size: *mut usize) -> ov_status_e;
}

// =============================================================================
// LLMPipeline
// =============================================================================

// NOTE: ov_genai_llm_pipeline_create is variadic and cannot be represented in the link! macro.
// Instead we provide a non-variadic wrapper that passes 0 properties.

unsafe extern "C" {
    #[doc = " Release the memory allocated by ov_genai_llm_pipeline."]
    pub fn ov_genai_llm_pipeline_free(pipe: *mut ov_genai_llm_pipeline);
}
unsafe extern "C" {
    #[doc = " Generate results by ov_genai_llm_pipeline."]
    pub fn ov_genai_llm_pipeline_generate(pipe: *mut ov_genai_llm_pipeline, inputs: *const ::std::os::raw::c_char, config: *const ov_genai_generation_config, streamer: *const streamer_callback, results: *mut *mut ov_genai_decoded_results) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Generate results by ov_genai_llm_pipeline using ChatHistory."]
    pub fn ov_genai_llm_pipeline_generate_with_history(pipe: *mut ov_genai_llm_pipeline, history: *const ov_genai_chat_history, config: *const ov_genai_generation_config, streamer: *const streamer_callback, results: *mut *mut ov_genai_decoded_results) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Start chat with keeping history in kv cache."]
    pub fn ov_genai_llm_pipeline_start_chat(pipe: *mut ov_genai_llm_pipeline) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Finish chat and clear kv cache."]
    pub fn ov_genai_llm_pipeline_finish_chat(pipe: *mut ov_genai_llm_pipeline) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Get the GenerationConfig from ov_genai_llm_pipeline."]
    pub fn ov_genai_llm_pipeline_get_generation_config(pipe: *const ov_genai_llm_pipeline, config: *mut *mut ov_genai_generation_config) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Set the GenerationConfig to ov_genai_llm_pipeline."]
    pub fn ov_genai_llm_pipeline_set_generation_config(pipe: *mut ov_genai_llm_pipeline, config: *mut ov_genai_generation_config) -> ov_status_e;
}

// =============================================================================
// GenerationConfig
// =============================================================================

unsafe extern "C" {
    #[doc = " Create ov_genai_generation_config."]
    pub fn ov_genai_generation_config_create(config: *mut *mut ov_genai_generation_config) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Create ov_genai_generation_config from JSON file."]
    pub fn ov_genai_generation_config_create_from_json(json_path: *const ::std::os::raw::c_char, config: *mut *mut ov_genai_generation_config) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Release the memory allocated by ov_genai_generation_config."]
    pub fn ov_genai_generation_config_free(handle: *mut ov_genai_generation_config);
}
unsafe extern "C" {
    #[doc = " Set the maximum number of tokens to generate."]
    pub fn ov_genai_generation_config_set_max_new_tokens(handle: *mut ov_genai_generation_config, value: usize) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Set the maximum length the generated tokens can have."]
    pub fn ov_genai_generation_config_set_max_length(config: *mut ov_genai_generation_config, value: usize) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Set whether or not to ignore <eos> token."]
    pub fn ov_genai_generation_config_set_ignore_eos(config: *mut ov_genai_generation_config, value: bool) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Set the minimum number of tokens to generate."]
    pub fn ov_genai_generation_config_set_min_new_tokens(config: *mut ov_genai_generation_config, value: usize) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Set whether or not to include user prompt in the output."]
    pub fn ov_genai_generation_config_set_echo(config: *mut ov_genai_generation_config, value: bool) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Set the number of top logprobs computed for each position."]
    pub fn ov_genai_generation_config_set_logprobs(config: *mut ov_genai_generation_config, value: usize) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Set the set of strings that will cause pipeline to stop generating."]
    pub fn ov_genai_generation_config_set_stop_strings(config: *mut ov_genai_generation_config, strings: *const *const ::std::os::raw::c_char, count: usize) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Set whether or not to include stop string in output."]
    pub fn ov_genai_generation_config_set_include_stop_str_in_output(config: *mut ov_genai_generation_config, value: bool) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Set the set of token ids that will cause pipeline to stop generating."]
    pub fn ov_genai_generation_config_set_stop_token_ids(config: *mut ov_genai_generation_config, token_ids: *const i64, token_ids_num: usize) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Set the number of beam groups for diverse beam search."]
    pub fn ov_genai_generation_config_set_num_beam_groups(config: *mut ov_genai_generation_config, value: usize) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Set the number of beams for beam search."]
    pub fn ov_genai_generation_config_set_num_beams(config: *mut ov_genai_generation_config, value: usize) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Set the diversity penalty for beam search."]
    pub fn ov_genai_generation_config_set_diversity_penalty(config: *mut ov_genai_generation_config, value: f32) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Set the length penalty for beam search."]
    pub fn ov_genai_generation_config_set_length_penalty(config: *mut ov_genai_generation_config, value: f32) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Set the number of sequences to return for beam search."]
    pub fn ov_genai_generation_config_set_num_return_sequences(config: *mut ov_genai_generation_config, value: usize) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Set the no_repeat_ngram_size."]
    pub fn ov_genai_generation_config_set_no_repeat_ngram_size(config: *mut ov_genai_generation_config, value: usize) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Set the stopping condition for grouped beam search."]
    pub fn ov_genai_generation_config_set_stop_criteria(config: *mut ov_genai_generation_config, value: StopCriteria) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Set the temperature value for random sampling."]
    pub fn ov_genai_generation_config_set_temperature(config: *mut ov_genai_generation_config, value: f32) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Set the top_p value for nucleus sampling."]
    pub fn ov_genai_generation_config_set_top_p(config: *mut ov_genai_generation_config, value: f32) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Set the top_k value for top-k filtering."]
    pub fn ov_genai_generation_config_set_top_k(config: *mut ov_genai_generation_config, value: usize) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Set whether to use multinomial random sampling."]
    pub fn ov_genai_generation_config_set_do_sample(config: *mut ov_genai_generation_config, value: bool) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Set the repetition penalty. 1.0 means no penalty."]
    pub fn ov_genai_generation_config_set_repetition_penalty(config: *mut ov_genai_generation_config, value: f32) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Set the presence penalty."]
    pub fn ov_genai_generation_config_set_presence_penalty(config: *mut ov_genai_generation_config, value: f32) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Set the frequency penalty."]
    pub fn ov_genai_generation_config_set_frequency_penalty(config: *mut ov_genai_generation_config, value: f32) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Set the seed for random number generator."]
    pub fn ov_genai_generation_config_set_rng_seed(config: *mut ov_genai_generation_config, value: usize) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Set the assistant confidence threshold for speculative decoding."]
    pub fn ov_genai_generation_config_set_assistant_confidence_threshold(config: *mut ov_genai_generation_config, value: f32) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Set the number of assistant tokens for speculative decoding."]
    pub fn ov_genai_generation_config_set_num_assistant_tokens(config: *mut ov_genai_generation_config, value: usize) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Set the maximum ngram size for prompt lookup."]
    pub fn ov_genai_generation_config_set_max_ngram_size(config: *mut ov_genai_generation_config, value: usize) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Set the eos token id."]
    pub fn ov_genai_generation_config_set_eos_token_id(config: *mut ov_genai_generation_config, id: i64) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Get the maximum number of tokens to generate."]
    pub fn ov_genai_generation_config_get_max_new_tokens(config: *const ov_genai_generation_config, max_new_tokens: *mut usize) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Validate the generation configuration."]
    pub fn ov_genai_generation_config_validate(config: *mut ov_genai_generation_config) -> ov_status_e;
}

// =============================================================================
// ChatHistory
// =============================================================================

unsafe extern "C" {
    #[doc = " Create a new empty ChatHistory instance."]
    pub fn ov_genai_chat_history_create(history: *mut *mut ov_genai_chat_history) -> ov_genai_chat_history_status_e;
}
unsafe extern "C" {
    #[doc = " Create a ChatHistory from a JsonContainer."]
    pub fn ov_genai_chat_history_create_from_json_container(history: *mut *mut ov_genai_chat_history, messages: *const ov_genai_json_container) -> ov_genai_chat_history_status_e;
}
unsafe extern "C" {
    #[doc = " Release the memory allocated by ov_genai_chat_history."]
    pub fn ov_genai_chat_history_free(history: *mut ov_genai_chat_history);
}
unsafe extern "C" {
    #[doc = " Add a message to the chat history from a JsonContainer."]
    pub fn ov_genai_chat_history_push_back(history: *mut ov_genai_chat_history, message: *const ov_genai_json_container) -> ov_genai_chat_history_status_e;
}
unsafe extern "C" {
    #[doc = " Remove the last message from the chat history."]
    pub fn ov_genai_chat_history_pop_back(history: *mut ov_genai_chat_history) -> ov_genai_chat_history_status_e;
}
unsafe extern "C" {
    #[doc = " Get all messages as a JsonContainer."]
    pub fn ov_genai_chat_history_get_messages(history: *const ov_genai_chat_history, messages: *mut *mut ov_genai_json_container) -> ov_genai_chat_history_status_e;
}
unsafe extern "C" {
    #[doc = " Get a message at a specific index."]
    pub fn ov_genai_chat_history_get_message(history: *const ov_genai_chat_history, index: usize, message: *mut *mut ov_genai_json_container) -> ov_genai_chat_history_status_e;
}
unsafe extern "C" {
    #[doc = " Get the first message."]
    pub fn ov_genai_chat_history_get_first(history: *const ov_genai_chat_history, message: *mut *mut ov_genai_json_container) -> ov_genai_chat_history_status_e;
}
unsafe extern "C" {
    #[doc = " Get the last message."]
    pub fn ov_genai_chat_history_get_last(history: *const ov_genai_chat_history, message: *mut *mut ov_genai_json_container) -> ov_genai_chat_history_status_e;
}
unsafe extern "C" {
    #[doc = " Clear all messages from the chat history."]
    pub fn ov_genai_chat_history_clear(history: *mut ov_genai_chat_history) -> ov_genai_chat_history_status_e;
}
unsafe extern "C" {
    #[doc = " Get the number of messages in the chat history."]
    pub fn ov_genai_chat_history_size(history: *const ov_genai_chat_history, size: *mut usize) -> ov_genai_chat_history_status_e;
}
unsafe extern "C" {
    #[doc = " Check if the chat history is empty."]
    pub fn ov_genai_chat_history_empty(history: *const ov_genai_chat_history, empty: *mut ::std::os::raw::c_int) -> ov_genai_chat_history_status_e;
}
unsafe extern "C" {
    #[doc = " Set tools definitions for function calling."]
    pub fn ov_genai_chat_history_set_tools(history: *mut ov_genai_chat_history, tools: *const ov_genai_json_container) -> ov_genai_chat_history_status_e;
}
unsafe extern "C" {
    #[doc = " Get tools definitions."]
    pub fn ov_genai_chat_history_get_tools(history: *const ov_genai_chat_history, tools: *mut *mut ov_genai_json_container) -> ov_genai_chat_history_status_e;
}
unsafe extern "C" {
    #[doc = " Set extra context for custom template variables."]
    pub fn ov_genai_chat_history_set_extra_context(history: *mut ov_genai_chat_history, extra_context: *const ov_genai_json_container) -> ov_genai_chat_history_status_e;
}
unsafe extern "C" {
    #[doc = " Get extra context."]
    pub fn ov_genai_chat_history_get_extra_context(history: *const ov_genai_chat_history, extra_context: *mut *mut ov_genai_json_container) -> ov_genai_chat_history_status_e;
}

// =============================================================================
// JsonContainer
// =============================================================================

unsafe extern "C" {
    #[doc = " Create a new empty JsonContainer instance."]
    pub fn ov_genai_json_container_create(container: *mut *mut ov_genai_json_container) -> ov_genai_json_container_status_e;
}
unsafe extern "C" {
    #[doc = " Create a JsonContainer from a JSON string."]
    pub fn ov_genai_json_container_create_from_json_string(container: *mut *mut ov_genai_json_container, json_str: *const ::std::os::raw::c_char) -> ov_genai_json_container_status_e;
}
unsafe extern "C" {
    #[doc = " Create a JsonContainer as an empty JSON object."]
    pub fn ov_genai_json_container_create_object(container: *mut *mut ov_genai_json_container) -> ov_genai_json_container_status_e;
}
unsafe extern "C" {
    #[doc = " Create a JsonContainer as an empty JSON array."]
    pub fn ov_genai_json_container_create_array(container: *mut *mut ov_genai_json_container) -> ov_genai_json_container_status_e;
}
unsafe extern "C" {
    #[doc = " Release the memory allocated by ov_genai_json_container."]
    pub fn ov_genai_json_container_free(container: *mut ov_genai_json_container);
}
unsafe extern "C" {
    #[doc = " Convert JsonContainer to JSON string."]
    pub fn ov_genai_json_container_to_json_string(container: *const ov_genai_json_container, output: *mut ::std::os::raw::c_char, output_size: *mut usize) -> ov_genai_json_container_status_e;
}
unsafe extern "C" {
    #[doc = " Create a copy of JsonContainer."]
    pub fn ov_genai_json_container_copy(source: *const ov_genai_json_container, target: *mut *mut ov_genai_json_container) -> ov_genai_json_container_status_e;
}

// =============================================================================
// VLMPipeline
// =============================================================================

// NOTE: ov_genai_vlm_pipeline_create is variadic and cannot be represented in the link! macro.

unsafe extern "C" {
    #[doc = " Release the memory allocated by ov_genai_vlm_pipeline."]
    pub fn ov_genai_vlm_pipeline_free(pipe: *mut ov_genai_vlm_pipeline);
}
unsafe extern "C" {
    #[doc = " Generate results by ov_genai_vlm_pipeline with text and image inputs."]
    pub fn ov_genai_vlm_pipeline_generate(pipe: *mut ov_genai_vlm_pipeline, text_inputs: *const ::std::os::raw::c_char, rgbs: *const *const ov_tensor_t, num_images: usize, config: *const ov_genai_generation_config, streamer: *const streamer_callback, results: *mut *mut ov_genai_vlm_decoded_results) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Start chat with keeping history in kv cache (VLM)."]
    pub fn ov_genai_vlm_pipeline_start_chat(pipe: *mut ov_genai_vlm_pipeline) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Finish chat and clear kv cache (VLM)."]
    pub fn ov_genai_vlm_pipeline_finish_chat(pipe: *mut ov_genai_vlm_pipeline) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Get the GenerationConfig from ov_genai_vlm_pipeline."]
    pub fn ov_genai_vlm_pipeline_get_generation_config(pipe: *const ov_genai_vlm_pipeline, config: *mut *mut ov_genai_generation_config) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Set the GenerationConfig to ov_genai_vlm_pipeline."]
    pub fn ov_genai_vlm_pipeline_set_generation_config(pipe: *mut ov_genai_vlm_pipeline, config: *mut ov_genai_generation_config) -> ov_status_e;
}

// VLM decoded results
unsafe extern "C" {
    #[doc = " Create VLMDecodedResults."]
    pub fn ov_genai_vlm_decoded_results_create(results: *mut *mut ov_genai_vlm_decoded_results) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Release the memory allocated by ov_genai_vlm_decoded_results."]
    pub fn ov_genai_vlm_decoded_results_free(results: *mut ov_genai_vlm_decoded_results);
}
unsafe extern "C" {
    #[doc = " Get performance metrics from ov_genai_vlm_decoded_results."]
    pub fn ov_genai_vlm_decoded_results_get_perf_metrics(results: *const ov_genai_vlm_decoded_results, metrics: *mut *mut ov_genai_perf_metrics) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Release performance metrics from VLM decoded results."]
    pub fn ov_genai_vlm_decoded_results_perf_metrics_free(metrics: *mut ov_genai_perf_metrics);
}
unsafe extern "C" {
    #[doc = " Get string result from ov_genai_vlm_decoded_results."]
    pub fn ov_genai_vlm_decoded_results_get_string(results: *const ov_genai_vlm_decoded_results, output: *mut ::std::os::raw::c_char, output_size: *mut usize) -> ov_status_e;
}

// =============================================================================
// WhisperPipeline
// =============================================================================

// NOTE: ov_genai_whisper_pipeline_create is variadic and cannot be represented in the link! macro.

unsafe extern "C" {
    #[doc = " Release the memory allocated by ov_genai_whisper_pipeline."]
    pub fn ov_genai_whisper_pipeline_free(pipeline: *mut ov_genai_whisper_pipeline);
}
unsafe extern "C" {
    #[doc = " Generate results by ov_genai_whisper_pipeline from raw speech input."]
    pub fn ov_genai_whisper_pipeline_generate(pipeline: *mut ov_genai_whisper_pipeline, raw_speech: *const f32, raw_speech_size: usize, config: *const ov_genai_whisper_generation_config, results: *mut *mut ov_genai_whisper_decoded_results) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Get the WhisperGenerationConfig from ov_genai_whisper_pipeline."]
    pub fn ov_genai_whisper_pipeline_get_generation_config(pipeline: *const ov_genai_whisper_pipeline, config: *mut *mut ov_genai_whisper_generation_config) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Set the WhisperGenerationConfig to ov_genai_whisper_pipeline."]
    pub fn ov_genai_whisper_pipeline_set_generation_config(pipeline: *mut ov_genai_whisper_pipeline, config: *mut ov_genai_whisper_generation_config) -> ov_status_e;
}

// =============================================================================
// WhisperGenerationConfig
// =============================================================================

unsafe extern "C" {
    #[doc = " Create ov_genai_whisper_generation_config."]
    pub fn ov_genai_whisper_generation_config_create(config: *mut *mut ov_genai_whisper_generation_config) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Create ov_genai_whisper_generation_config from JSON file."]
    pub fn ov_genai_whisper_generation_config_create_from_json(json_path: *const ::std::os::raw::c_char, config: *mut *mut ov_genai_whisper_generation_config) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Release the memory allocated by ov_genai_whisper_generation_config."]
    pub fn ov_genai_whisper_generation_config_free(config: *mut ov_genai_whisper_generation_config);
}
unsafe extern "C" {
    #[doc = " Get the underlying GenerationConfig from WhisperGenerationConfig."]
    pub fn ov_genai_whisper_generation_config_get_generation_config(config: *const ov_genai_whisper_generation_config, generation_config: *mut *mut ov_genai_generation_config) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Set the decoder start token id."]
    pub fn ov_genai_whisper_generation_config_set_decoder_start_token_id(config: *mut ov_genai_whisper_generation_config, token_id: i64) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Get the decoder start token id."]
    pub fn ov_genai_whisper_generation_config_get_decoder_start_token_id(config: *const ov_genai_whisper_generation_config, token_id: *mut i64) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Set the padding token id."]
    pub fn ov_genai_whisper_generation_config_set_pad_token_id(config: *mut ov_genai_whisper_generation_config, token_id: i64) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Get the padding token id."]
    pub fn ov_genai_whisper_generation_config_get_pad_token_id(config: *const ov_genai_whisper_generation_config, token_id: *mut i64) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Set the translate token id."]
    pub fn ov_genai_whisper_generation_config_set_translate_token_id(config: *mut ov_genai_whisper_generation_config, token_id: i64) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Get the translate token id."]
    pub fn ov_genai_whisper_generation_config_get_translate_token_id(config: *const ov_genai_whisper_generation_config, token_id: *mut i64) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Set the transcribe token id."]
    pub fn ov_genai_whisper_generation_config_set_transcribe_token_id(config: *mut ov_genai_whisper_generation_config, token_id: i64) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Get the transcribe token id."]
    pub fn ov_genai_whisper_generation_config_get_transcribe_token_id(config: *const ov_genai_whisper_generation_config, token_id: *mut i64) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Set the previous start of transcript token id."]
    pub fn ov_genai_whisper_generation_config_set_prev_sot_token_id(config: *mut ov_genai_whisper_generation_config, token_id: i64) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Get the previous start of transcript token id."]
    pub fn ov_genai_whisper_generation_config_get_prev_sot_token_id(config: *const ov_genai_whisper_generation_config, token_id: *mut i64) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Set the no timestamps token id."]
    pub fn ov_genai_whisper_generation_config_set_no_timestamps_token_id(config: *mut ov_genai_whisper_generation_config, token_id: i64) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Get the no timestamps token id."]
    pub fn ov_genai_whisper_generation_config_get_no_timestamps_token_id(config: *const ov_genai_whisper_generation_config, token_id: *mut i64) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Set the maximum initial timestamp index."]
    pub fn ov_genai_whisper_generation_config_set_max_initial_timestamp_index(config: *mut ov_genai_whisper_generation_config, index: usize) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Get the maximum initial timestamp index."]
    pub fn ov_genai_whisper_generation_config_get_max_initial_timestamp_index(config: *const ov_genai_whisper_generation_config, index: *mut usize) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Set whether the model is multilingual."]
    pub fn ov_genai_whisper_generation_config_set_is_multilingual(config: *mut ov_genai_whisper_generation_config, is_multilingual: bool) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Get whether the model is multilingual."]
    pub fn ov_genai_whisper_generation_config_get_is_multilingual(config: *const ov_genai_whisper_generation_config, is_multilingual: *mut bool) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Set the language for generation."]
    pub fn ov_genai_whisper_generation_config_set_language(config: *mut ov_genai_whisper_generation_config, language: *const ::std::os::raw::c_char) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Get the language for generation."]
    pub fn ov_genai_whisper_generation_config_get_language(config: *const ov_genai_whisper_generation_config, language: *mut ::std::os::raw::c_char, language_size: *mut usize) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Set the task for generation."]
    pub fn ov_genai_whisper_generation_config_set_task(config: *mut ov_genai_whisper_generation_config, task: *const ::std::os::raw::c_char) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Get the task for generation."]
    pub fn ov_genai_whisper_generation_config_get_task(config: *const ov_genai_whisper_generation_config, task: *mut ::std::os::raw::c_char, task_size: *mut usize) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Set whether to return timestamps."]
    pub fn ov_genai_whisper_generation_config_set_return_timestamps(config: *mut ov_genai_whisper_generation_config, return_timestamps: bool) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Get whether to return timestamps."]
    pub fn ov_genai_whisper_generation_config_get_return_timestamps(config: *const ov_genai_whisper_generation_config, return_timestamps: *mut bool) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Set the initial prompt for generation."]
    pub fn ov_genai_whisper_generation_config_set_initial_prompt(config: *mut ov_genai_whisper_generation_config, initial_prompt: *const ::std::os::raw::c_char) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Get the initial prompt for generation."]
    pub fn ov_genai_whisper_generation_config_get_initial_prompt(config: *const ov_genai_whisper_generation_config, initial_prompt: *mut ::std::os::raw::c_char, prompt_size: *mut usize) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Set the hotwords for generation."]
    pub fn ov_genai_whisper_generation_config_set_hotwords(config: *mut ov_genai_whisper_generation_config, hotwords: *const ::std::os::raw::c_char) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Get the hotwords for generation."]
    pub fn ov_genai_whisper_generation_config_get_hotwords(config: *const ov_genai_whisper_generation_config, hotwords: *mut ::std::os::raw::c_char, hotwords_size: *mut usize) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Set the begin suppress tokens."]
    pub fn ov_genai_whisper_generation_config_set_begin_suppress_tokens(config: *mut ov_genai_whisper_generation_config, tokens: *const i64, tokens_count: usize) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Get the begin suppress tokens count."]
    pub fn ov_genai_whisper_generation_config_get_begin_suppress_tokens_count(config: *const ov_genai_whisper_generation_config, tokens_count: *mut usize) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Get the begin suppress tokens."]
    pub fn ov_genai_whisper_generation_config_get_begin_suppress_tokens(config: *const ov_genai_whisper_generation_config, tokens: *mut i64, tokens_count: usize) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Set the suppress tokens."]
    pub fn ov_genai_whisper_generation_config_set_suppress_tokens(config: *mut ov_genai_whisper_generation_config, tokens: *const i64, tokens_count: usize) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Get the suppress tokens count."]
    pub fn ov_genai_whisper_generation_config_get_suppress_tokens_count(config: *const ov_genai_whisper_generation_config, tokens_count: *mut usize) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Get the suppress tokens."]
    pub fn ov_genai_whisper_generation_config_get_suppress_tokens(config: *const ov_genai_whisper_generation_config, tokens: *mut i64, tokens_count: usize) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Validate the whisper generation configuration."]
    pub fn ov_genai_whisper_generation_config_validate(config: *mut ov_genai_whisper_generation_config) -> ov_status_e;
}

// =============================================================================
// Whisper decoded results
// =============================================================================

unsafe extern "C" {
    #[doc = " Create WhisperDecodedResultChunk."]
    pub fn ov_genai_whisper_decoded_result_chunk_create(chunk: *mut *mut ov_genai_whisper_decoded_result_chunk) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Release the memory allocated by ov_genai_whisper_decoded_result_chunk."]
    pub fn ov_genai_whisper_decoded_result_chunk_free(chunk: *mut ov_genai_whisper_decoded_result_chunk);
}
unsafe extern "C" {
    #[doc = " Get start timestamp from a whisper decoded result chunk."]
    pub fn ov_genai_whisper_decoded_result_chunk_get_start_ts(chunk: *const ov_genai_whisper_decoded_result_chunk, start_ts: *mut f32) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Get end timestamp from a whisper decoded result chunk."]
    pub fn ov_genai_whisper_decoded_result_chunk_get_end_ts(chunk: *const ov_genai_whisper_decoded_result_chunk, end_ts: *mut f32) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Get text from a whisper decoded result chunk."]
    pub fn ov_genai_whisper_decoded_result_chunk_get_text(chunk: *const ov_genai_whisper_decoded_result_chunk, text: *mut ::std::os::raw::c_char, text_size: *mut usize) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Create WhisperDecodedResults."]
    pub fn ov_genai_whisper_decoded_results_create(results: *mut *mut ov_genai_whisper_decoded_results) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Release the memory allocated by ov_genai_whisper_decoded_results."]
    pub fn ov_genai_whisper_decoded_results_free(results: *mut ov_genai_whisper_decoded_results);
}
unsafe extern "C" {
    #[doc = " Get performance metrics from ov_genai_whisper_decoded_results."]
    pub fn ov_genai_whisper_decoded_results_get_perf_metrics(results: *const ov_genai_whisper_decoded_results, metrics: *mut *mut ov_genai_perf_metrics) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Get number of text results from whisper decoded results."]
    pub fn ov_genai_whisper_decoded_results_get_texts_count(results: *const ov_genai_whisper_decoded_results, count: *mut usize) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Get text result at specific index from whisper decoded results."]
    pub fn ov_genai_whisper_decoded_results_get_text_at(results: *const ov_genai_whisper_decoded_results, index: usize, text: *mut ::std::os::raw::c_char, text_size: *mut usize) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Get score at specific index from whisper decoded results."]
    pub fn ov_genai_whisper_decoded_results_get_score_at(results: *const ov_genai_whisper_decoded_results, index: usize, score: *mut f32) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Check if chunks are available from whisper decoded results."]
    pub fn ov_genai_whisper_decoded_results_has_chunks(results: *const ov_genai_whisper_decoded_results, has_chunks: *mut bool) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Get number of chunks from whisper decoded results."]
    pub fn ov_genai_whisper_decoded_results_get_chunks_count(results: *const ov_genai_whisper_decoded_results, count: *mut usize) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Get chunk at specific index from whisper decoded results."]
    pub fn ov_genai_whisper_decoded_results_get_chunk_at(results: *const ov_genai_whisper_decoded_results, index: usize, chunk: *mut *mut ov_genai_whisper_decoded_result_chunk) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Get string representation from whisper decoded results."]
    pub fn ov_genai_whisper_decoded_results_get_string(results: *const ov_genai_whisper_decoded_results, output: *mut ::std::os::raw::c_char, output_size: *mut usize) -> ov_status_e;
}

// =============================================================================
// PerfMetrics
// =============================================================================

unsafe extern "C" {
    #[doc = " Get load time from perf metrics."]
    pub fn ov_genai_perf_metrics_get_load_time(metrics: *const ov_genai_perf_metrics, load_time: *mut f32) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Get the number of generated tokens."]
    pub fn ov_genai_perf_metrics_get_num_generation_tokens(metrics: *const ov_genai_perf_metrics, num_generation_tokens: *mut usize) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Get the number of input tokens."]
    pub fn ov_genai_perf_metrics_get_num_input_tokens(metrics: *const ov_genai_perf_metrics, num_input_tokens: *mut usize) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Get time to first token (mean and std in ms)."]
    pub fn ov_genai_perf_metrics_get_ttft(metrics: *const ov_genai_perf_metrics, mean: *mut f32, std: *mut f32) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Get time per output token (mean and std in ms)."]
    pub fn ov_genai_perf_metrics_get_tpot(metrics: *const ov_genai_perf_metrics, mean: *mut f32, std: *mut f32) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Get inference time per output token (mean and std in ms)."]
    pub fn ov_genai_perf_metrics_get_ipot(metrics: *const ov_genai_perf_metrics, mean: *mut f32, std: *mut f32) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Get tokens per second (mean and std)."]
    pub fn ov_genai_perf_metrics_get_throughput(metrics: *const ov_genai_perf_metrics, mean: *mut f32, std: *mut f32) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Get inference duration (mean and std in ms)."]
    pub fn ov_genai_perf_metrics_get_inference_duration(metrics: *const ov_genai_perf_metrics, mean: *mut f32, std: *mut f32) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Get generate duration (mean and std in ms)."]
    pub fn ov_genai_perf_metrics_get_generate_duration(metrics: *const ov_genai_perf_metrics, mean: *mut f32, std: *mut f32) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Get tokenization duration (mean and std in ms)."]
    pub fn ov_genai_perf_metrics_get_tokenization_duration(metrics: *const ov_genai_perf_metrics, mean: *mut f32, std: *mut f32) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Get detokenization duration (mean and std in ms)."]
    pub fn ov_genai_perf_metrics_get_detokenization_duration(metrics: *const ov_genai_perf_metrics, mean: *mut f32, std: *mut f32) -> ov_status_e;
}
unsafe extern "C" {
    #[doc = " Add perf metrics from right to left in place."]
    pub fn ov_genai_perf_metrics_add_in_place(left: *mut ov_genai_perf_metrics, right: *const ov_genai_perf_metrics) -> ov_status_e;
}

} // link!
