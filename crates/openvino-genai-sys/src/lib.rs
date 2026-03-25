//! This crate provides low-level, unsafe, Rust bindings to OpenVINO™ GenAI using its [C API]. If
//! you are looking to use OpenVINO™ GenAI from Rust, you likely should look at the ergonomic, safe
//! bindings in [openvino-genai], which depends on this crate. See the repository [README] for more
//! information, including build instructions.
//!
//! [C API]: https://github.com/openvinotoolkit/openvino.genai
//! [openvino-genai-sys]: https://crates.io/crates/openvino-genai-sys
//! [openvino-genai]: https://crates.io/crates/openvino-genai
//! [README]: https://github.com/intel/openvino-rs/tree/main/crates/openvino-genai-sys
//!
//! An example interaction with raw [openvino-genai-sys]:
//! ```ignore
//! openvino_genai_sys::library::load().expect("to have an OpenVINO GenAI library available");
//! ```

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]
#![allow(unused, dead_code)]
#![deny(clippy::all)]
#![warn(clippy::pedantic)]
#![warn(clippy::cargo)]
#![allow(
    clippy::must_use_candidate,
    clippy::suspicious_doc_comments,
    clippy::wildcard_imports,
    clippy::doc_markdown
)]

mod linking;

mod generated;
pub use generated::*;

// Re-export shared types from openvino-sys so that users of both crates share a single definition.
pub use openvino_sys::ov_status_e;
pub use openvino_sys::ov_tensor_t;

/// Contains extra utilities for finding and loading the OpenVINO GenAI shared libraries.
pub mod library {
    use std::path::PathBuf;

    /// When compiled with the `runtime-linking` feature, load the function definitions from a
    /// shared library; with the `dynamic-linking` feature, this function does nothing since the
    /// library has already been linked.
    ///
    /// # Errors
    ///
    /// When compiled with the `runtime-linking` feature, this may fail if the `openvino-finder`
    /// cannot discover the library on the current system.
    pub fn load() -> Result<(), String> {
        super::generated::load()?;
        init_variadic(find().as_deref())
    }

    /// Load the OpenVINO GenAI shared library from an explicit path.
    ///
    /// This is useful when the library is located in a non-standard directory that cannot be
    /// discovered by the `openvino-finder` search paths or environment variables — for example,
    /// when the path is read from a configuration file.
    ///
    /// The `path` should point to the `openvino_genai_c` shared library file
    /// (e.g., `libopenvino_genai_c.so`).
    ///
    /// # Errors
    ///
    /// May fail if the shared library cannot be opened or is invalid.
    pub fn load_from(path: impl Into<std::path::PathBuf>) -> Result<(), String> {
        let path = path.into();
        super::generated::load_from(path.clone())?;
        init_variadic(Some(&path))
    }

    /// Initialize the variadic pipeline creation functions from the loaded library.
    #[allow(unused_variables)]
    fn init_variadic(path: Option<&std::path::Path>) -> Result<(), String> {
        #[cfg(feature = "runtime-linking")]
        if let Some(path) = path {
            super::runtime_variadic::init_variadic_fns(path)?;
        }
        Ok(())
    }

    /// Return the location of the shared library `openvino-genai-sys` will link to. If compiled
    /// with runtime linking, this will attempt to discover the location of an `openvino_genai_c`
    /// shared library on the system. Otherwise (with dynamic linking or compilation from source),
    /// this relies on a static path discovered at build time.
    ///
    /// Knowing the location of the OpenVINO GenAI libraries can be useful for ensuring the correct
    /// runtime environment is configured.
    pub fn find() -> Option<PathBuf> {
        if cfg!(feature = "runtime-linking") {
            openvino_finder::find("openvino_genai_c", openvino_finder::Linking::Dynamic)
        } else {
            Some(PathBuf::from(env!("OPENVINO_GENAI_LIB_PATH")))
        }
    }
}

// The variadic pipeline creation functions cannot go through the link! macro. We provide them as
// direct extern declarations (only available with dynamic linking) or as helper functions.

#[cfg(feature = "dynamic-linking")]
extern "C" {
    /// Construct ov_genai_llm_pipeline with no additional properties.
    ///
    /// This is the variadic C function; call with `property_args_size = 0` and no trailing args.
    pub fn ov_genai_llm_pipeline_create(
        models_path: *const ::std::os::raw::c_char,
        device: *const ::std::os::raw::c_char,
        property_args_size: usize,
        pipe: *mut *mut ov_genai_llm_pipeline,
        ...
    ) -> ov_status_e;

    /// Construct ov_genai_vlm_pipeline with no additional properties.
    pub fn ov_genai_vlm_pipeline_create(
        models_path: *const ::std::os::raw::c_char,
        device: *const ::std::os::raw::c_char,
        property_args_size: usize,
        pipe: *mut *mut ov_genai_vlm_pipeline,
        ...
    ) -> ov_status_e;

    /// Construct ov_genai_whisper_pipeline with no additional properties.
    pub fn ov_genai_whisper_pipeline_create(
        models_path: *const ::std::os::raw::c_char,
        device: *const ::std::os::raw::c_char,
        property_args_size: usize,
        pipeline: *mut *mut ov_genai_whisper_pipeline,
        ...
    ) -> ov_status_e;
}

// For runtime linking, we load these functions manually since the link! macro can't handle variadics.
// These use OnceLock so they can be initialized either lazily (from `find()`) or explicitly (from
// `load_from`). The `library::load()` and `library::load_from()` functions call
// `init_variadic_fns` to populate them.
#[cfg(feature = "runtime-linking")]
mod runtime_variadic {
    use super::*;
    use std::path::Path;
    use std::sync::OnceLock;

    type CreateFn = unsafe extern "C" fn(
        *const ::std::os::raw::c_char,
        *const ::std::os::raw::c_char,
        usize,
        *mut *mut ::std::os::raw::c_void,
    ) -> ov_status_e;

    static LLM_CREATE: OnceLock<CreateFn> = OnceLock::new();
    static VLM_CREATE: OnceLock<CreateFn> = OnceLock::new();
    static WHISPER_CREATE: OnceLock<CreateFn> = OnceLock::new();

    /// Initialize the variadic pipeline creation functions from the library at `path`.
    ///
    /// Called internally by `library::load()` and `library::load_from()`.
    pub(crate) fn init_variadic_fns(path: &Path) -> Result<(), String> {
        unsafe {
            let lib = libloading::Library::new(path).map_err(|e| {
                format!(
                    "failed to open shared library for variadic fns at {}: {}",
                    path.display(),
                    e,
                )
            })?;

            if let Ok(sym) = lib.get::<CreateFn>(b"ov_genai_llm_pipeline_create") {
                let _ = LLM_CREATE.set(*sym);
            }
            if let Ok(sym) = lib.get::<CreateFn>(b"ov_genai_vlm_pipeline_create") {
                let _ = VLM_CREATE.set(*sym);
            }
            if let Ok(sym) = lib.get::<CreateFn>(b"ov_genai_whisper_pipeline_create") {
                let _ = WHISPER_CREATE.set(*sym);
            }

            // Leak the library to keep function pointers valid.
            std::mem::forget(lib);
        }
        Ok(())
    }

    /// Create an LLM pipeline with no additional properties (runtime-linking variant).
    ///
    /// # Safety
    ///
    /// The caller must ensure that `models_path` and `device` are valid C strings, and `pipe` is
    /// a valid pointer to receive the created pipeline.
    ///
    /// # Panics
    ///
    /// Panics if `library::load()` or `library::load_from()` has not been called first.
    pub unsafe fn ov_genai_llm_pipeline_create(
        models_path: *const ::std::os::raw::c_char,
        device: *const ::std::os::raw::c_char,
        property_args_size: usize,
        pipe: *mut *mut ov_genai_llm_pipeline,
    ) -> ov_status_e {
        let f = LLM_CREATE
            .get()
            .expect("`openvino_genai_c` function not loaded: `ov_genai_llm_pipeline_create`; call library::load() or library::load_from() first");
        f(models_path, device, property_args_size, pipe.cast())
    }

    /// Create a VLM pipeline with no additional properties (runtime-linking variant).
    ///
    /// # Safety
    ///
    /// Same safety requirements as [`ov_genai_llm_pipeline_create`].
    ///
    /// # Panics
    ///
    /// Panics if `library::load()` or `library::load_from()` has not been called first.
    pub unsafe fn ov_genai_vlm_pipeline_create(
        models_path: *const ::std::os::raw::c_char,
        device: *const ::std::os::raw::c_char,
        property_args_size: usize,
        pipe: *mut *mut ov_genai_vlm_pipeline,
    ) -> ov_status_e {
        let f = VLM_CREATE
            .get()
            .expect("`openvino_genai_c` function not loaded: `ov_genai_vlm_pipeline_create`; call library::load() or library::load_from() first");
        f(models_path, device, property_args_size, pipe.cast())
    }

    /// Create a Whisper pipeline with no additional properties (runtime-linking variant).
    ///
    /// # Safety
    ///
    /// Same safety requirements as [`ov_genai_llm_pipeline_create`].
    ///
    /// # Panics
    ///
    /// Panics if `library::load()` or `library::load_from()` has not been called first.
    pub unsafe fn ov_genai_whisper_pipeline_create(
        models_path: *const ::std::os::raw::c_char,
        device: *const ::std::os::raw::c_char,
        property_args_size: usize,
        pipeline: *mut *mut ov_genai_whisper_pipeline,
    ) -> ov_status_e {
        let f = WHISPER_CREATE
            .get()
            .expect("`openvino_genai_c` function not loaded: `ov_genai_whisper_pipeline_create`; call library::load() or library::load_from() first");
        f(models_path, device, property_args_size, pipeline.cast())
    }
}

#[cfg(feature = "runtime-linking")]
pub use runtime_variadic::{
    ov_genai_llm_pipeline_create, ov_genai_vlm_pipeline_create,
    ov_genai_whisper_pipeline_create,
};
