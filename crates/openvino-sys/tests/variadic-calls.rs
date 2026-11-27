//! Exercise the C-variadic functions of the OpenVINO C API at run time.
//!
//! The sibling `variadic.rs` checks the variadic bindings with the type system alone: that a
//! binding keeps its `...`. That catches a lost `...` at compile time, but it cannot catch a
//! symbol that fails to resolve, nor prove that the varargs actually arrive at the callee. This
//! file does both, so it needs a working OpenVINO installation.
//!
//! A corrupted variadic call can still return `OK`, so every check here reads the value back
//! rather than trusting the status code.

use openvino_sys::*;
use std::ffi::{CStr, CString};

/// A tiny ONNX identity model, enough to compile for CPU. Shared with the `openvino` crate tests.
const ONNX_MODEL: &[u8] = b"\x08\x07\x12\nonnx-wally:j\n*\n\x06inputs\x12\x07outputs\x1a\ridentity_node\"\x08Identity\x12\x0bno-op-modelZ\x16\n\x06inputs\x12\x0c\n\n\x08\x01\x12\x06\n\x00\n\x02\x08\x02b\x17\n\x07outputs\x12\x0c\n\n\x08\x01\x12\x06\n\x00\n\x02\x08\x02B\x02\x10\x0c";

/// An NHWC model, needed by the pre-processing checks: `set_color_format` rejects a shape it
/// cannot interpret as an image.
const IR_MODEL_NHWC: &str = r#"<?xml version="1.0"?>
<net name="m" version="11">
  <layers>
    <layer id="0" name="inputs" type="Parameter" version="opset1">
      <data shape="1,224,224,3" element_type="f32"/>
      <output>
        <port id="0" precision="FP32" names="inputs">
          <dim>1</dim><dim>224</dim><dim>224</dim><dim>3</dim>
        </port>
      </output>
    </layer>
    <layer id="1" name="outputs" type="Result" version="opset1">
      <input>
        <port id="0" precision="FP32">
          <dim>1</dim><dim>224</dim><dim>224</dim><dim>3</dim>
        </port>
      </input>
    </layer>
  </layers>
  <edges>
    <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
  </edges>
</net>
"#;

/// The null pointer that terminates a variadic argument list.
const END: *const std::os::raw::c_char = std::ptr::null();

fn core() -> *mut ov_core_t {
    openvino_sys::library::load().expect("an OpenVINO library to be available");
    let mut core = std::ptr::null_mut();
    assert_eq!(unsafe { ov_core_create(&mut core) }, ov_status_e::OK);
    core
}

fn read_onnx(core: *mut ov_core_t) -> *mut ov_model_t {
    let mut model = std::ptr::null_mut();
    assert_eq!(
        unsafe {
            ov_core_read_model_from_memory_buffer(
                core,
                ONNX_MODEL.as_ptr().cast(),
                ONNX_MODEL.len(),
                std::ptr::null(),
                &mut model,
            )
        },
        ov_status_e::OK
    );
    model
}

fn read_ir_nhwc(core: *mut ov_core_t) -> *mut ov_model_t {
    let mut model = std::ptr::null_mut();
    assert_eq!(
        unsafe {
            ov_core_read_model_from_memory_buffer(
                core,
                IR_MODEL_NHWC.as_ptr().cast(),
                IR_MODEL_NHWC.len(),
                std::ptr::null(),
                &mut model,
            )
        },
        ov_status_e::OK
    );
    model
}

fn core_property(core: *mut ov_core_t, key: &str) -> String {
    let device = CString::new("CPU").unwrap();
    let key = CString::new(key).unwrap();
    let mut value = std::ptr::null_mut();
    assert_eq!(
        unsafe { ov_core_get_property(core, device.as_ptr(), key.as_ptr(), &mut value) },
        ov_status_e::OK
    );
    let owned = unsafe { CStr::from_ptr(value) }
        .to_string_lossy()
        .into_owned();
    unsafe { ov_free(value) };
    owned
}

fn compiled_property(compiled: *mut ov_compiled_model_t, key: &str) -> String {
    let key = CString::new(key).unwrap();
    let mut value = std::ptr::null_mut();
    assert_eq!(
        unsafe { ov_compiled_model_get_property(compiled, key.as_ptr(), &mut value) },
        ov_status_e::OK,
        "reading back a property must succeed"
    );
    let owned = unsafe { CStr::from_ptr(value) }
        .to_string_lossy()
        .into_owned();
    unsafe { ov_free(value) };
    owned
}

fn last_error() -> String {
    let message = unsafe { ov_get_last_err_msg() };
    if message.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(message) }
            .to_string_lossy()
            .into_owned()
    }
}

/// Resolve the address behind each variadic binding.
///
/// With `runtime-linking`, `link_variadic!()` binds each symbol to a `LazyLock` that looks it up in
/// the shared library on first use; a name that is missing or misspelled panics *there*, not at
/// compile time, so nothing but an actual call (or this deref) proves the symbol exists. With
/// `dynamic-linking` the resolution already happened at link time and taking the address merely
/// confirms it.
macro_rules! address_of {
    ($name:ident) => {{
        #[cfg(feature = "runtime-linking")]
        let address = *$name as *const () as usize;
        #[cfg(not(feature = "runtime-linking"))]
        let address = $name as *const () as usize;
        (stringify!($name), address)
    }};
}

/// Every variadic symbol must resolve in the loaded library, including the ones no test can call
/// for lack of the hardware they need (the remote-context family requires a GPU).
#[test]
fn every_variadic_symbol_resolves() {
    openvino_sys::library::load().expect("an OpenVINO library to be available");
    let symbols = [
        address_of!(ov_remote_context_create_tensor),
        address_of!(ov_compiled_model_set_property),
        address_of!(ov_core_compile_model),
        address_of!(ov_core_compile_model_from_file),
        address_of!(ov_core_compile_model_from_file_unicode),
        address_of!(ov_core_set_property),
        address_of!(ov_core_create_context),
        address_of!(ov_core_compile_model_with_context),
        address_of!(ov_preprocess_input_tensor_info_set_color_format_with_subname),
    ];
    for (name, address) in symbols {
        assert_ne!(address, 0, "`{name}` resolved to a null address");
    }
}

/// `ov_core_set_property()` must receive both the key and the value pointer intact. Reading the
/// value back is the actual check: a call made with the wrong convention can still return `OK`.
#[test]
fn core_set_property_round_trips() {
    let core = core();
    let device = CString::new("CPU").unwrap();
    for value in ["THROUGHPUT", "LATENCY"] {
        let key = CString::new("PERFORMANCE_HINT").unwrap();
        let value_cstr = CString::new(value).unwrap();
        assert_eq!(
            unsafe {
                ov_core_set_property(
                    core,
                    device.as_ptr(),
                    key.as_ptr(),
                    value_cstr.as_ptr(),
                    END,
                )
            },
            ov_status_e::OK,
            "setting PERFORMANCE_HINT={value}"
        );
        assert_eq!(core_property(core, "PERFORMANCE_HINT"), value);
    }
    unsafe { ov_core_free(core) };
}

/// A long value exercises the pointer itself rather than a small integer that might survive a
/// wrong convention by luck, and `CACHE_DIR` is stored verbatim instead of being normalised to an
/// enum, so any truncation shows up in the read-back.
#[test]
fn core_set_property_passes_a_long_value_intact() {
    let core = core();
    let device = CString::new("CPU").unwrap();
    let directory = std::env::temp_dir().join("openvino-rs-variadic-".repeat(8));
    let expected = directory.to_string_lossy().into_owned();

    let key = CString::new("CACHE_DIR").unwrap();
    let value = CString::new(expected.clone()).unwrap();
    assert_eq!(
        unsafe { ov_core_set_property(core, device.as_ptr(), key.as_ptr(), value.as_ptr(), END) },
        ov_status_e::OK
    );
    assert_eq!(core_property(core, "CACHE_DIR"), expected);
    unsafe { ov_core_free(core) };
}

/// `ov_core_set_property()` takes a `const char* device_name` that may be null, which means "set
/// this globally". The null sits in a *fixed* parameter, immediately before the varargs, so it is
/// exactly the spot where a convention mismatch would misalign the rest of the list.
#[test]
fn core_set_property_accepts_a_null_device_name() {
    let core = core();
    let directory = std::env::temp_dir().join("openvino-rs-variadic-global");
    let expected = directory.to_string_lossy().into_owned();

    let key = CString::new("CACHE_DIR").unwrap();
    let value = CString::new(expected.clone()).unwrap();
    assert_eq!(
        unsafe { ov_core_set_property(core, std::ptr::null(), key.as_ptr(), value.as_ptr(), END) },
        ov_status_e::OK
    );
    // A global property is visible through the CPU device.
    assert_eq!(core_property(core, "CACHE_DIR"), expected);
    unsafe { ov_core_free(core) };
}

/// An unknown key must come back as an error that *quotes the key we passed*. That is direct
/// evidence the vararg pointer arrived intact: the message could not name the key otherwise.
///
/// Note the asymmetry with [`compile_model_rejects_an_unknown_property_key`]: `ov_core_set_property()`
/// hands the key to the plugin lazily, so an unknown key is accepted here and only rejected once a
/// model is compiled.
#[test]
fn core_set_property_defers_an_unknown_key_to_compile_time() {
    let core = core();
    let device = CString::new("CPU").unwrap();
    let key = CString::new("NOT_A_REAL_PROPERTY").unwrap();
    let value = CString::new("whatever").unwrap();
    assert_eq!(
        unsafe { ov_core_set_property(core, device.as_ptr(), key.as_ptr(), value.as_ptr(), END) },
        ov_status_e::OK,
        "an unknown key is not validated by `ov_core_set_property()` itself"
    );

    // Compiling is where the plugin sees it, and the error names the key we passed.
    let model = read_onnx(core);
    let mut compiled = std::ptr::null_mut();
    let status = unsafe { ov_core_compile_model(core, model, device.as_ptr(), 0, &mut compiled) };
    assert_ne!(status, ov_status_e::OK);
    assert!(
        last_error().contains("NOT_A_REAL_PROPERTY"),
        "the error should quote the key that was passed through the varargs, got: {}",
        last_error()
    );
    unsafe { ov_model_free(model) };
    unsafe { ov_core_free(core) };
}

/// `ov_core_set_property()` consumes exactly **one** `<key, value>` pair, despite the plural wording
/// in the docs. `GET_PROPERTY_FROM_ARGS_LIST` reads one key and one value with no loop, so a
/// second pair in the same call is silently dropped.
///
/// This is pinned down because it is the reason `Core::set_properties` has to loop in Rust and call
/// the C function once per property: batching them into a single call would quietly lose all but
/// the first.
#[test]
fn core_set_property_consumes_only_the_first_pair() {
    let core = core();
    let device = CString::new("CPU").unwrap();
    let first_key = CString::new("INFERENCE_NUM_THREADS").unwrap();
    let first_value = CString::new("3").unwrap();
    let second_key = CString::new("PERFORMANCE_HINT").unwrap();
    let second_value = CString::new("THROUGHPUT").unwrap();

    let hint_before = core_property(core, "PERFORMANCE_HINT");
    assert_eq!(
        unsafe {
            ov_core_set_property(
                core,
                device.as_ptr(),
                first_key.as_ptr(),
                first_value.as_ptr(),
                second_key.as_ptr(),
                second_value.as_ptr(),
                END,
            )
        },
        ov_status_e::OK
    );

    assert_eq!(
        core_property(core, "INFERENCE_NUM_THREADS"),
        "3",
        "the first pair is applied"
    );
    assert_eq!(
        core_property(core, "PERFORMANCE_HINT"),
        hint_before,
        "the second pair is ignored, so the hint must be unchanged"
    );
    unsafe { ov_core_free(core) };
}

/// `ov_core_compile_model` takes its properties through the varargs and, unlike `set_property()`,
/// genuinely loops over them. Until now every call in this repository passed zero of them, so the
/// variadic part of this signature went completely unexercised.
///
/// Four varargs is also where a wrong convention bites hardest on AArch64: the function has five
/// fixed parameters, so under the fixed-arity declaration the first varargs would travel in
/// registers while the callee reads all of them from the stack.
#[test]
fn compile_model_applies_properties_from_varargs() {
    let core = core();
    let model = read_onnx(core);
    let device = CString::new("CPU").unwrap();
    let hint_key = CString::new("PERFORMANCE_HINT").unwrap();
    let hint_value = CString::new("THROUGHPUT").unwrap();
    let threads_key = CString::new("INFERENCE_NUM_THREADS").unwrap();
    let threads_value = CString::new("3").unwrap();

    let mut compiled = std::ptr::null_mut();
    // `property_args_size` counts arguments, not pairs: two pairs is four.
    assert_eq!(
        unsafe {
            ov_core_compile_model(
                core,
                model,
                device.as_ptr(),
                4,
                &mut compiled,
                hint_key.as_ptr(),
                hint_value.as_ptr(),
                threads_key.as_ptr(),
                threads_value.as_ptr(),
            )
        },
        ov_status_e::OK,
        "compiling with two property pairs: {}",
        last_error()
    );

    assert_eq!(
        compiled_property(compiled, "PERFORMANCE_HINT"),
        "THROUGHPUT"
    );
    assert_eq!(compiled_property(compiled, "INFERENCE_NUM_THREADS"), "3");

    unsafe { ov_compiled_model_free(compiled) };
    unsafe { ov_model_free(model) };
    unsafe { ov_core_free(core) };
}

/// `property_args_size` must be even, and the guard rejecting an odd count runs *before* any
/// vararg is read.
#[test]
fn compile_model_rejects_an_odd_property_count() {
    let core = core();
    let model = read_onnx(core);
    let device = CString::new("CPU").unwrap();
    let key = CString::new("PERFORMANCE_HINT").unwrap();
    let mut compiled = std::ptr::null_mut();
    assert_eq!(
        unsafe {
            ov_core_compile_model(core, model, device.as_ptr(), 3, &mut compiled, key.as_ptr())
        },
        ov_status_e::INVALID_C_PARAM
    );
    unsafe { ov_model_free(model) };
    unsafe { ov_core_free(core) };
}

/// An unknown key passed through `ov_core_compile_model()`'s varargs is rejected, and
/// the message quotes the key (proof the pointer survived the call).
#[test]
fn compile_model_rejects_an_unknown_property_key() {
    let core = core();
    let model = read_onnx(core);
    let device = CString::new("CPU").unwrap();
    let key = CString::new("NOT_A_REAL_PROPERTY").unwrap();
    let value = CString::new("whatever").unwrap();
    let mut compiled = std::ptr::null_mut();
    let status = unsafe {
        ov_core_compile_model(
            core,
            model,
            device.as_ptr(),
            2,
            &mut compiled,
            key.as_ptr(),
            value.as_ptr(),
        )
    };
    assert_ne!(status, ov_status_e::OK);
    assert!(
        last_error().contains("NOT_A_REAL_PROPERTY"),
        "the error should quote the key passed through the varargs, got: {}",
        last_error()
    );
    unsafe { ov_model_free(model) };
    unsafe { ov_core_free(core) };
}

/// The same check for `ov_core_compile_model_from_file()`, whose varargs were also never exercised.
#[test]
fn compile_model_from_file_applies_properties_from_varargs() {
    let core = core();
    let directory = std::env::temp_dir().join(format!(
        "openvino-rs-variadic-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.join("model.onnx");
    std::fs::write(&path, ONNX_MODEL).unwrap();

    let path_cstr = CString::new(path.to_str().unwrap()).unwrap();
    let device = CString::new("CPU").unwrap();
    let key = CString::new("PERFORMANCE_HINT").unwrap();
    let value = CString::new("THROUGHPUT").unwrap();
    let mut compiled = std::ptr::null_mut();
    let status = unsafe {
        ov_core_compile_model_from_file(
            core,
            path_cstr.as_ptr(),
            device.as_ptr(),
            2,
            &mut compiled,
            key.as_ptr(),
            value.as_ptr(),
        )
    };
    std::fs::remove_dir_all(&directory).ok();

    assert_eq!(status, ov_status_e::OK, "{}", last_error());
    assert_eq!(
        compiled_property(compiled, "PERFORMANCE_HINT"),
        "THROUGHPUT"
    );

    unsafe { ov_compiled_model_free(compiled) };
    unsafe { ov_core_free(core) };
}

/// `ov_preprocess_input_tensor_info_set_color_format_with_subname()` is the strongest available
/// check: its vararg loop reads `sub_names_size` strings, and each one becomes part of an input
/// name in the built model. Asserting on those names proves every vararg pointer arrived intact
/// and in order (not just that the call did not crash).
#[test]
fn preprocess_set_color_format_passes_every_subname() {
    let core = core();
    let model = read_ir_nhwc(core);

    let mut pipeline = std::ptr::null_mut();
    assert_eq!(
        unsafe { ov_preprocess_prepostprocessor_create(model, &mut pipeline) },
        ov_status_e::OK
    );
    let mut input_info = std::ptr::null_mut();
    assert_eq!(
        unsafe { ov_preprocess_prepostprocessor_get_input_info(pipeline, &mut input_info) },
        ov_status_e::OK
    );
    let mut tensor_info = std::ptr::null_mut();
    assert_eq!(
        unsafe { ov_preprocess_input_info_get_tensor_info(input_info, &mut tensor_info) },
        ov_status_e::OK
    );

    let y_plane = CString::new("y_plane").unwrap();
    let uv_plane = CString::new("uv_plane").unwrap();
    assert_eq!(
        unsafe {
            ov_preprocess_input_tensor_info_set_color_format_with_subname(
                tensor_info,
                ov_color_format_e::NV12_TWO_PLANES,
                2,
                y_plane.as_ptr(),
                uv_plane.as_ptr(),
            )
        },
        ov_status_e::OK,
        "{}",
        last_error()
    );

    // Converting back to BGR lets the pipeline build; NV12 input alone cannot feed an FP32 model.
    let mut steps = std::ptr::null_mut();
    assert_eq!(
        unsafe { ov_preprocess_input_info_get_preprocess_steps(input_info, &mut steps) },
        ov_status_e::OK
    );
    assert_eq!(
        unsafe { ov_preprocess_preprocess_steps_convert_color(steps, ov_color_format_e::BGR) },
        ov_status_e::OK
    );

    let mut built = std::ptr::null_mut();
    assert_eq!(
        unsafe { ov_preprocess_prepostprocessor_build(pipeline, &mut built) },
        ov_status_e::OK,
        "{}",
        last_error()
    );

    // The single input became two planes, each named after one of the subnames we passed.
    let mut inputs = 0usize;
    assert_eq!(
        unsafe { ov_model_inputs_size(built, &mut inputs) },
        ov_status_e::OK
    );
    assert_eq!(inputs, 2, "NV12_TWO_PLANES should split the input in two");

    let mut names = Vec::new();
    for index in 0..inputs {
        let mut port = std::ptr::null_mut();
        assert_eq!(
            unsafe { ov_model_const_input_by_index(built, index, &mut port) },
            ov_status_e::OK
        );
        let mut name = std::ptr::null_mut();
        assert_eq!(
            unsafe { ov_port_get_any_name(port, &mut name) },
            ov_status_e::OK
        );
        names.push(
            unsafe { CStr::from_ptr(name) }
                .to_string_lossy()
                .into_owned(),
        );
        unsafe { ov_free(name) };
    }
    assert_eq!(names, ["inputs/y_plane", "inputs/uv_plane"]);

    unsafe { ov_model_free(built) };
    unsafe { ov_preprocess_prepostprocessor_free(pipeline) };
    unsafe { ov_model_free(model) };
    unsafe { ov_core_free(core) };
}

/// With `sub_names_size` of zero the variadic list is empty and must not be touched at all.
#[test]
fn preprocess_set_color_format_accepts_no_subnames() {
    let core = core();
    let model = read_ir_nhwc(core);

    let mut pipeline = std::ptr::null_mut();
    assert_eq!(
        unsafe { ov_preprocess_prepostprocessor_create(model, &mut pipeline) },
        ov_status_e::OK
    );
    let mut input_info = std::ptr::null_mut();
    assert_eq!(
        unsafe { ov_preprocess_prepostprocessor_get_input_info(pipeline, &mut input_info) },
        ov_status_e::OK
    );
    let mut tensor_info = std::ptr::null_mut();
    assert_eq!(
        unsafe { ov_preprocess_input_info_get_tensor_info(input_info, &mut tensor_info) },
        ov_status_e::OK
    );
    assert_eq!(
        unsafe {
            ov_preprocess_input_tensor_info_set_color_format_with_subname(
                tensor_info,
                ov_color_format_e::BGR,
                0,
            )
        },
        ov_status_e::OK
    );

    unsafe { ov_preprocess_prepostprocessor_free(pipeline) };
    unsafe { ov_model_free(model) };
    unsafe { ov_core_free(core) };
}
