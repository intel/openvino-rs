#[doc(hidden)]
#[macro_export]
macro_rules! link {
    (
        $(
            extern "C" {
                $(#[doc=$doc:expr])*
                $(#[cfg($cfg:meta)])*
                pub fn $name:ident($($pname:ident: $pty:ty),* $(,)?) $(-> $ret:ty)*;
            }
        )+
    ) => (
        /// When compiled as a dynamically-linked library, this function does nothing. It exists to
        /// provide a consistent API with the runtime-linked version.
        pub fn load() -> Result<(), String> {
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
