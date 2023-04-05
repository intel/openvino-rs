use thiserror::Error;

/// Enumerate errors returned by the OpenVINO implementation. See
/// [`IEStatusCode`](https://docs.openvinotoolkit.org/latest/ie_c_api/ie__c__api_8h.html#a391683b1e8e26df8b58d7033edd9ee83).
// TODO This could be auto-generated (https://github.com/intel/openvino-rs/issues/20).
#[allow(missing_docs)]
#[derive(Debug, Error, PartialEq, Eq)]
pub enum InferenceError {
    #[error("general error")]
    GeneralError,
    #[error("not implemented")]
    NotImplemented,
    #[error("network not loaded")]
    NetworkNotLoaded,
    #[error("parameter mismatch")]
    ParameterMismatch,
    #[error("not found")]
    NotFound,
    #[error("out of bounds")]
    OutOfBounds,
    #[error("unexpected")]
    Unexpected,
    #[error("request busy")]
    RequestBusy,
    #[error("result not ready")]
    ResultNotReady,
    #[error("not allocated")]
    NotAllocated,
    #[error("infer not started")]
    InferNotStarted,
    #[error("network not ready")]
    NetworkNotReady,
    #[error("undefined error code: {0}")]
    Undefined(i32),
}

impl InferenceError {
    /// Convert an `error_code` to a [`Result`]:
    /// - `0` becomes `Ok`
    /// - anything else becomes `Err` containing an [`InferenceError`]
    pub fn from(error_code: i32) -> Result<(), InferenceError> {
        #[allow(clippy::enum_glob_use)]
        use InferenceError::*;
        match error_code {
            openvino_sys::IEStatusCode_OK => Ok(()),
            openvino_sys::IEStatusCode_GENERAL_ERROR => Err(GeneralError),
            openvino_sys::IEStatusCode_NOT_IMPLEMENTED => Err(NotImplemented),
            openvino_sys::IEStatusCode_NETWORK_NOT_LOADED => Err(NetworkNotLoaded),
            openvino_sys::IEStatusCode_PARAMETER_MISMATCH => Err(ParameterMismatch),
            openvino_sys::IEStatusCode_NOT_FOUND => Err(NotFound),
            openvino_sys::IEStatusCode_OUT_OF_BOUNDS => Err(OutOfBounds),
            openvino_sys::IEStatusCode_UNEXPECTED => Err(Unexpected),
            openvino_sys::IEStatusCode_REQUEST_BUSY => Err(RequestBusy),
            openvino_sys::IEStatusCode_RESULT_NOT_READY => Err(ResultNotReady),
            openvino_sys::IEStatusCode_NOT_ALLOCATED => Err(NotAllocated),
            openvino_sys::IEStatusCode_INFER_NOT_STARTED => Err(InferNotStarted),
            openvino_sys::IEStatusCode_NETWORK_NOT_READ => Err(NetworkNotReady),
            _ => Err(Undefined(error_code)),
        }
    }
}

/// Enumerate setup failures: in some cases, this library will call library-loading code that may
/// fail in a different way (i.e., [`LoadingError`]) than the calls to the OpenVINO libraries (i.e.,
/// [`InferenceError`]).
#[allow(missing_docs)]
#[derive(Debug, Error)]
pub enum SetupError {
    #[error("inference error")]
    Inference(#[from] InferenceError),
    #[error("library loading error")]
    Loading(#[from] LoadingError),
}

/// Enumerate the ways that library loading can fail.
#[allow(missing_docs)]
#[derive(Debug, Error)]
pub enum LoadingError {
    #[error("system failed to load shared libraries (see https://github.com/intel/openvino-rs/blob/main/crates/openvino-finder): {0}")]
    SystemFailure(String),
    #[error("cannot find path to shared libraries (see https://github.com/intel/openvino-rs/blob/main/crates/openvino-finder)")]
    CannotFindLibraryPath,
    #[error("cannot find path to XML plugin configuration (see https://github.com/intel/openvino-rs/blob/main/crates/openvino-finder)")]
    CannotFindPluginPath,
    #[error("unable to convert path to a UTF-8 string (see https://doc.rust-lang.org/std/path/struct.Path.html#method.to_str)")]
    CannotStringifyPath,
}
