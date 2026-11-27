//! Property round-trips through the safe wrappers over the C-variadic property setters.
//!
//! `Core::set_property()` and `CompiledModel::set_property()` both call a C-variadic function,
//! which these integration test cover.
//!
//! `crates/openvino-sys/tests/variadic-calls.rs` covers the raw C calls; this file covers the safe
//! wrappers. Each check reads the value back: a variadic call
//! made with the wrong convention can still report success.

use openvino::{Core, DeviceType, PropertyKey, RwPropertyKey};

/// A tiny ONNX identity model. Enough to compile for CPU without downloading a fixture.
const ONNX_MODEL: &[u8] = b"\x08\x07\x12\nonnx-wally:j\n*\n\x06inputs\x12\x07outputs\x1a\ridentity_node\"\x08Identity\x12\x0bno-op-modelZ\x16\n\x06inputs\x12\x0c\n\n\x08\x01\x12\x06\n\x00\n\x02\x08\x02b\x17\n\x07outputs\x12\x0c\n\n\x08\x01\x12\x06\n\x00\n\x02\x08\x02B\x02\x10\x0c";

/// `Core::set_property()` must round-trip, and must keep working when called repeatedly: the
/// `runtime-linking` build resolves the symbol once into a `LazyLock`, so a second call goes
/// through the cached function pointer rather than the lookup path.
#[test]
fn core_set_property_round_trips_repeatedly() {
    let mut core = Core::new().unwrap();
    for value in ["THROUGHPUT", "LATENCY", "THROUGHPUT"] {
        core.set_property(&DeviceType::CPU, &RwPropertyKey::HintPerformanceMode, value)
            .unwrap_or_else(|e| panic!("failed to set PERFORMANCE_HINT={value}: {e}"));
        let read_back = core
            .get_property(
                &DeviceType::CPU,
                &PropertyKey::Rw(RwPropertyKey::HintPerformanceMode),
            )
            .unwrap();
        assert_eq!(read_back, value, "PERFORMANCE_HINT did not round-trip");
    }
}

/// `Core::set_properties()` applies *every* pair it is given.
///
/// This is worth pinning down because the underlying C function consumes only a single
/// `<key, value>` pair per call (`GET_PROPERTY_FROM_ARGS_LIST` reads one key and one value, with no
/// loop), so the wrapper has to call it once per property. If it were ever "optimized" into a
/// single variadic call with all the pairs, every property but the first would be silently dropped
/// and a status-only assertion would not notice.
#[test]
fn core_set_properties_applies_every_pair() {
    let mut core = Core::new().unwrap();
    let cache_dir = std::env::temp_dir().join("openvino-rs-set-properties");
    let cache_dir = cache_dir.to_string_lossy().into_owned();

    core.set_properties(
        &DeviceType::CPU,
        [
            (RwPropertyKey::HintPerformanceMode, "THROUGHPUT"),
            (RwPropertyKey::InferenceNumThreads, "3"),
            (RwPropertyKey::CacheDir, cache_dir.as_str()),
        ],
    )
    .unwrap();

    for (key, expected) in [
        (RwPropertyKey::HintPerformanceMode, "THROUGHPUT"),
        (RwPropertyKey::InferenceNumThreads, "3"),
        (RwPropertyKey::CacheDir, cache_dir.as_str()),
    ] {
        let read_back = core
            .get_property(&DeviceType::CPU, &PropertyKey::Rw(key.clone()))
            .unwrap_or_else(|e| panic!("failed to read back {key:?}: {e}"));
        assert_eq!(read_back, expected, "{key:?} did not round-trip");
    }
}

/// A property value long enough that it cannot ride along in a register by accident, and stored
/// verbatim rather than normalized to an enum, so truncation or corruption is visible.
#[test]
fn core_set_property_passes_a_long_value_intact() {
    let mut core = Core::new().unwrap();
    let directory = std::env::temp_dir().join("openvino-rs-long-value-".repeat(8));
    let expected = directory.to_string_lossy().into_owned();

    core.set_property(&DeviceType::CPU, &RwPropertyKey::CacheDir, &expected)
        .unwrap();
    let read_back = core
        .get_property(&DeviceType::CPU, &PropertyKey::Rw(RwPropertyKey::CacheDir))
        .unwrap();
    assert_eq!(read_back, expected);
}

/// Test that `CompiledModel::set_property()` round trips
///
/// It is tested through `BATCH:CPU` on purpose. The CPU plugin refuses to set any property on an
/// already-compiled model, so against plain `CPU` this call can only ever return an error.
/// The auto-batch plugin does accept `AUTO_BATCH_TIMEOUT`, so the value
/// can be read back and the pointers proven to have survived the call.
#[test]
fn compiled_model_set_property_round_trips() {
    let mut core = Core::new().unwrap();
    let model = core.read_model_from_buffer(ONNX_MODEL, None).unwrap();
    let mut compiled = core
        .compile_model(&model, DeviceType::Other("BATCH:CPU".into()))
        .unwrap();

    for value in ["2000", "500"] {
        compiled
            .set_property(&RwPropertyKey::AutoBatchTimeout, value)
            .unwrap_or_else(|e| panic!("failed to set AUTO_BATCH_TIMEOUT={value}: {e}"));
        let read_back = compiled
            .get_property(&PropertyKey::Rw(RwPropertyKey::AutoBatchTimeout))
            .unwrap();
        assert_eq!(read_back, value, "AUTO_BATCH_TIMEOUT did not round-trip");
    }
}

/// Properties passed to `compile_model()` are applied to the resulting compiled model.
///
/// `Core::compile_model()` calls `ov_core_compile_model()`, which is variadic, and passes zero
/// properties through the varargs. This checks the compiled model still reports a property set on
/// the `Core` beforehand, which is the only route the safe API offers today.
#[test]
fn compiled_model_inherits_a_core_property() {
    let mut core = Core::new().unwrap();
    core.set_property(
        &DeviceType::CPU,
        &RwPropertyKey::HintPerformanceMode,
        "THROUGHPUT",
    )
    .unwrap();

    let model = core.read_model_from_buffer(ONNX_MODEL, None).unwrap();
    let compiled = core.compile_model(&model, DeviceType::CPU).unwrap();
    let read_back = compiled
        .get_property(&PropertyKey::Rw(RwPropertyKey::HintPerformanceMode))
        .unwrap();
    assert_eq!(read_back, "THROUGHPUT");
}

/// Setting a property must not disturb an unrelated one. A variadic call that walked off the end
/// of its argument list could corrupt neighboring state without failing outright.
#[test]
fn setting_one_property_leaves_others_alone() {
    let mut core = Core::new().unwrap();
    core.set_property(&DeviceType::CPU, &RwPropertyKey::InferenceNumThreads, "3")
        .unwrap();
    core.set_property(
        &DeviceType::CPU,
        &RwPropertyKey::HintPerformanceMode,
        "THROUGHPUT",
    )
    .unwrap();

    assert_eq!(
        core.get_property(
            &DeviceType::CPU,
            &PropertyKey::Rw(RwPropertyKey::InferenceNumThreads)
        )
        .unwrap(),
        "3",
        "INFERENCE_NUM_THREADS was clobbered by a later set_property"
    );
    assert_eq!(
        core.get_property(
            &DeviceType::CPU,
            &PropertyKey::Rw(RwPropertyKey::HintPerformanceMode)
        )
        .unwrap(),
        "THROUGHPUT"
    );
}

/// A property whose value contains an interior null byte must be rejected rather than silently
/// truncated: `cstr!()` builds a `CString`, which refuses it. Covers the `set_property()` path for the
/// null-byte handling fixed alongside the variadic work.
#[test]
#[should_panic(expected = "a valid C string")]
fn core_set_property_rejects_an_interior_null_byte() {
    let mut core = Core::new().unwrap();
    let _ = core.set_property(
        &DeviceType::CPU,
        &RwPropertyKey::CacheDir,
        "/tmp/with\0interior-null",
    );
}
