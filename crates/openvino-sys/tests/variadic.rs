//! Check that the C-variadic functions of the OpenVINO C API keep their variadic signature.
//!
//! These functions used to go through `link!`, which drops the `...` and declares them with a
//! fixed arity. That is an ABI mismatch: on targets where the variadic calling convention differs
//! from the fixed one (AArch64 macOS passes varargs on the stack rather than in registers), the
//! callee reads its variadic arguments from memory that was never written, and crashes.
//!
//! The checks are done by the type system, so they need no OpenVINO library at run time: if a
//! binding loses its `...`, this file fails to compile.

use openvino_sys::*;
use std::os::raw::c_char;

/// Assert at compile time that `$name` is bound as `unsafe extern "C" fn($($ty),*, ...) -> ov_status_e`.
///
/// With `runtime-linking`, variadic functions are `LazyLock` statics holding a function pointer
/// (see `link_variadic!`); with `dynamic-linking`, they are `extern` function items. Either way, a
/// fixed-arity binding does not coerce to the variadic function pointer type.
macro_rules! assert_variadic {
    ($name:ident($($ty:ty),* $(,)?)) => {{
        type Signature = unsafe extern "C" fn($($ty),* , ...) -> ov_status_e;
        #[cfg(feature = "runtime-linking")]
        let _: &std::sync::LazyLock<Signature> = &$name;
        #[cfg(not(feature = "runtime-linking"))]
        let _: Signature = $name;
        stringify!($name)
    }};
}

/// The variadic functions of the OpenVINO C API, with their fixed parameters.
fn checked_variadic_functions() -> Vec<&'static str> {
    vec![
        assert_variadic!(
            ov_remote_context_create_tensor(
                *const ov_remote_context_t,
                ov_element_type_e,
                ov_shape_t,
                usize,
                *mut *mut ov_tensor_t,
            )
        ),
        assert_variadic!(ov_compiled_model_set_property(*const ov_compiled_model_t)),
        assert_variadic!(
            ov_core_compile_model(
                *const ov_core_t,
                *const ov_model_t,
                *const c_char,
                usize,
                *mut *mut ov_compiled_model_t,
            )
        ),
        assert_variadic!(
            ov_core_compile_model_from_file(
                *const ov_core_t,
                *const c_char,
                *const c_char,
                usize,
                *mut *mut ov_compiled_model_t,
            )
        ),
        // `wchar_t` is an alias of `c_char` in the generated bindings.
        assert_variadic!(
            ov_core_compile_model_from_file_unicode(
                *const ov_core_t,
                *const c_char,
                *const c_char,
                usize,
                *mut *mut ov_compiled_model_t,
            )
        ),
        assert_variadic!(ov_core_set_property(*const ov_core_t, *const c_char)),
        assert_variadic!(
            ov_core_create_context(
                *const ov_core_t,
                *const c_char,
                usize,
                *mut *mut ov_remote_context_t,
            )
        ),
        assert_variadic!(
            ov_core_compile_model_with_context(
                *const ov_core_t,
                *const ov_model_t,
                *const ov_remote_context_t,
                usize,
                *mut *mut ov_compiled_model_t,
            )
        ),
        assert_variadic!(
            ov_preprocess_input_tensor_info_set_color_format_with_subname(
                *mut ov_preprocess_input_tensor_info_t,
                ov_color_format_e,
                usize,
            )
        ),
    ]
}

/// Names of the functions declared in the generated `link_variadic!` block.
fn generated_variadic_functions() -> Vec<&'static str> {
    const FUNCTIONS: &str = include_str!("../src/generated/functions.rs");
    let start = FUNCTIONS
        .find("crate::link_variadic! {")
        .expect("the generated bindings have no `link_variadic!` block");
    FUNCTIONS[start..]
        .lines()
        .filter_map(|line| line.trim().strip_prefix("pub fn "))
        .map(|rest| rest.split('(').next().unwrap())
        .collect()
}

/// Every variadic function in the generated bindings must be checked above, so that a function
/// added upstream does not go unverified (and vice versa, a check must not outlive its function).
#[test]
fn all_variadic_functions_are_checked() {
    let mut checked = checked_variadic_functions();
    let mut generated = generated_variadic_functions();
    checked.sort_unstable();
    generated.sort_unstable();
    assert_eq!(checked, generated);
}
