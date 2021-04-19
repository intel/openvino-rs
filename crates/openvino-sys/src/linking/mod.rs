//! This module determines how the OpenVINO libraries are linked to by defining a `link!` macro to
//! wrap the bindgen functions. In the `dynamic-linking` case, the macro is a pass-through; in the
//! `runtime-linking` case, the macro defines a `load` function to open the shared library and each
//! OpenVINO function is wrapped to use the loaded references.

#[cfg(feature = "dynamic-linking")]
mod dynamic;
// #[cfg(feature = "dynamic-linking")]
// pub use dynamic::link;

#[cfg(feature = "runtime-linking")]
mod runtime;
// #[cfg(feature = "runtime-linking")]
// pub use runtime::link;
