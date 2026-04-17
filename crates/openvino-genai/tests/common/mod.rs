/// Shared helpers for `crates/openvino-genai` integration tests.

/// Returns `true` if the OpenVINO GenAI runtime can be loaded on this machine.
/// If not, prints a message and returns `false` so the caller can early-return
/// from the test (effectively skipping it in CI environments without GenAI).
pub fn genai_available() -> bool {
    match openvino_genai::load() {
        Ok(()) => true,
        Err(e) => {
            eprintln!("Skipping test: OpenVINO GenAI runtime not available: {e:?}");
            false
        }
    }
}
