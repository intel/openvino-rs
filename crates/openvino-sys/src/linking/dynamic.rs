#[doc(hidden)]
#[macro_export]
macro_rules! link {
    (
        $(
            unsafe extern "C" {
                $(#[doc=$doc:expr])*
                $(#[cfg($cfg:meta)])*
                pub fn $name:ident($($pname:ident: $pty:ty),* $(,)?$(,...)?) $(-> $ret:ty)*;
            }
        )+
    ) => (
        /// When compiled as a dynamically-linked library, this function does nothing. It exists to
        /// provide a consistent API with the runtime-linked version.
        ///
        /// # Errors
        ///
        /// This version never fails.
        pub fn load() -> Result<(), String> {
            Ok(())
        }

        /// When compiled as a dynamically-linked library, this function does nothing. The library
        /// is already linked at compile time, so the `path` parameter is ignored. It exists to
        /// provide a consistent API with the runtime-linked version.
        ///
        /// # Errors
        ///
        /// This version never fails.
        pub fn load_from(_path: std::path::PathBuf) -> Result<(), String> {
            Ok(())
        }

        // Re-export all of the shared functions as-is.
        extern "C" {
            $(
                $(#[doc=$doc])*
                $(#[cfg($cfg)])*
                pub fn $name($($pname: $pty), *) $(-> $ret)*;
            )+
        }
    )
}

/// Bind C-variadic functions, mirroring [`link!`] for the `dynamic-linking` case.
///
/// The runtime-linking version of this macro cannot proxy variadic functions through a generated
/// Rust `fn` (Rust cannot *define* a C-variadic function on stable), so it binds them to function
/// pointers instead. When linking at compile time there is no such problem: the declarations can be
/// re-exported as-is, keeping the `...` so the correct calling convention is used.
#[doc(hidden)]
#[macro_export]
macro_rules! link_variadic {
    (
        $(
            unsafe extern "C" {
                $(#[doc=$doc:expr])*
                $(#[cfg($cfg:meta)])*
                pub fn $name:ident($($pname:ident: $pty:ty),* , ...) $(-> $ret:ty)*;
            }
        )+
    ) => (
        extern "C" {
            $(
                $(#[doc=$doc])*
                $(#[cfg($cfg)])*
                pub fn $name($($pname: $pty),* , ...) $(-> $ret)*;
            )+
        }
    )
}
