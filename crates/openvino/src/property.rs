use std::borrow::Cow;

/// `PropertyKey` represents valid configuration properties for a [crate::Core] instance.
#[derive(Debug)]
pub enum PropertyKey {
    /// Arbitrary string property key.
    Other(Cow<'static, str>),
}
