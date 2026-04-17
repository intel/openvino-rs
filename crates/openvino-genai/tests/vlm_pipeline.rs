//! Integration tests for VlmPipeline (requires OpenVINO GenAI runtime and model fixtures).

mod fixtures;

use fixtures::internvl2 as fixture;
use openvino_genai::VlmPipeline;

fn try_pipeline() -> Option<VlmPipeline> {
    let model_dir = fixture::model_dir();
    let model_path = model_dir.to_string_lossy();

    match VlmPipeline::new(&model_path, "CPU") {
        Ok(p) => Some(p),
        Err(e) => {
            eprintln!("Skipping GenAI VlmPipeline tests: failed to create pipeline: {e}");
            None
        }
    }
}

#[test]
fn test_create_pipeline() {
    let _pipeline = match try_pipeline() {
        Some(p) => p,
        None => return, // skip
    };
}
