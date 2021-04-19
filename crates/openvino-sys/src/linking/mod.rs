//! This module determines how the OpenVINO libraries are linked to by defining a `link!` macro to
//! wrap the bindgen functions. In the `dynamic-linking` case, the macro is a pass-through; in the
//! `runtime-linking` case, the macro defines a `load` function to open the shared library and each
//! OpenVINO function is wrapped to use the loaded references.
//!
//! This approach borrows heavily from the approach taken in
//! the `clang-sys` crate (see
//! [link.rs](https://github.com/KyleMayes/clang-sys/blob/c9ae24a7a218e73e1eccd320174349eef5a3bd1a/src/link.rs)).

#[cfg(feature = "dynamic-linking")]
mod dynamic;

#[cfg(feature = "runtime-linking")]
mod runtime;
