use openvino_sys::ov_status_e;
use std::error::Error;
use std::ffi::CStr;
use std::fmt;

/// See
/// [`ov_status_e`](https://docs.openvino.ai/2024/api/c_cpp_api/group__ov__base__c__api.html#_CPPv411ov_status_e);
/// enumerates errors returned by the OpenVINO implementation.
#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InferenceErrorKind {
    GeneralError,
    NotImplemented,
    NetworkNotLoaded,
    ParameterMismatch,
    NotFound,
    OutOfBounds,
    Unexpected,
    RequestBusy,
    ResultNotReady,
    NotAllocated,
    InferNotStarted,
    NetworkNotRead,
    InferCancelled,
    InvalidCParam,
    UnknownCError,
    NotImplementCMethod,
    UnknownException,
    Undefined(i32),
}

/// An error from OpenVINO operations, including the error kind and optional detailed message
/// from the C API.
#[derive(Debug, Clone, PartialEq)]
pub struct InferenceError {
    /// The kind of error that occurred
    pub kind: InferenceErrorKind,
    /// Detailed error message from OpenVINO C API, if available
    pub message: Option<String>,
}

impl InferenceError {
    /// Convert an `openvino_sys` error to a [`Result`]:
    /// - `0` becomes `Ok`
    /// - anything else becomes `Err` containing an [`InferenceError`]
    pub fn convert(status: ov_status_e, message: Option<String>) -> Result<(), InferenceError> {
        use InferenceErrorKind::{
            GeneralError, InferCancelled, InferNotStarted, InvalidCParam, NetworkNotLoaded,
            NetworkNotRead, NotAllocated, NotFound, NotImplementCMethod, NotImplemented,
            OutOfBounds, ParameterMismatch, RequestBusy, ResultNotReady, Unexpected, UnknownCError,
            UnknownException,
        };
        let kind = match status {
            ov_status_e::OK => return Ok(()),
            ov_status_e::GENERAL_ERROR => GeneralError,
            ov_status_e::NOT_IMPLEMENTED => NotImplemented,
            ov_status_e::NETWORK_NOT_LOADED => NetworkNotLoaded,
            ov_status_e::PARAMETER_MISMATCH => ParameterMismatch,
            ov_status_e::NOT_FOUND => NotFound,
            ov_status_e::OUT_OF_BOUNDS => OutOfBounds,
            ov_status_e::UNEXPECTED => Unexpected,
            ov_status_e::REQUEST_BUSY => RequestBusy,
            ov_status_e::RESULT_NOT_READY => ResultNotReady,
            ov_status_e::NOT_ALLOCATED => NotAllocated,
            ov_status_e::INFER_NOT_STARTED => InferNotStarted,
            ov_status_e::NETWORK_NOT_READ => NetworkNotRead,
            ov_status_e::INFER_CANCELLED => InferCancelled,
            ov_status_e::INVALID_C_PARAM => InvalidCParam,
            ov_status_e::UNKNOWN_C_ERROR => UnknownCError,
            ov_status_e::NOT_IMPLEMENT_C_METHOD => NotImplementCMethod,
            ov_status_e::UNKNOW_EXCEPTION => UnknownException,
        };
        Err(InferenceError { kind, message })
    }
}

impl Error for InferenceError {}

impl fmt::Display for InferenceErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GeneralError => write!(f, "general error"),
            Self::NotImplemented => write!(f, "not implemented"),
            Self::NetworkNotLoaded => write!(f, "network not loaded"),
            Self::ParameterMismatch => write!(f, "parameter mismatch"),
            Self::NotFound => write!(f, "not found"),
            Self::OutOfBounds => write!(f, "out of bounds"),
            Self::Unexpected => write!(f, "unexpected"),
            Self::RequestBusy => write!(f, "request busy"),
            Self::ResultNotReady => write!(f, "result not ready"),
            Self::NotAllocated => write!(f, "not allocated"),
            Self::InferNotStarted => write!(f, "infer not started"),
            Self::NetworkNotRead => write!(f, "network not read"),
            Self::InferCancelled => write!(f, "infer cancelled"),
            Self::InvalidCParam => write!(f, "invalid C parameter"),
            Self::UnknownCError => write!(f, "unknown C error"),
            Self::NotImplementCMethod => write!(f, "not implemented C method"),
            Self::UnknownException => write!(f, "unknown exception"),
            Self::Undefined(code) => write!(f, "undefined error code: {code}"),
        }
    }
}

impl fmt::Display for InferenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.kind)?;
        if let Some(msg) = &self.message {
            write!(f, ": {msg}")?;
        }
        Ok(())
    }
}

/// Enumerate the ways that library loading can fail.
#[allow(missing_docs)]
#[derive(Debug)]
pub enum LoadingError {
    SystemFailure(String),
    CannotFindLibraryPath,
    CannotFindPluginPath,
    CannotStringifyPath,
}

impl Error for LoadingError {}

impl fmt::Display for LoadingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SystemFailure(msg) => write!(f, "system failed to load shared libraries (see https://github.com/intel/openvino-rs/blob/main/crates/openvino-finder): {msg}"),
            Self::CannotFindLibraryPath => write!(f, "cannot find path to shared libraries (see https://github.com/intel/openvino-rs/blob/main/crates/openvino-finder)"),
            Self::CannotFindPluginPath => write!(f, "cannot find path to XML plugin configuration (see https://github.com/intel/openvino-rs/blob/main/crates/openvino-finder)"),
            Self::CannotStringifyPath => write!(f, "unable to convert path to a UTF-8 string (see https://doc.rust-lang.org/std/path/struct.Path.html#method.to_str)"),
        }
    }
}

/// Enumerate setup failures: in some cases, this library will call library-loading code that may
/// fail in a different way (i.e., [`LoadingError`]) than the calls to the OpenVINO libraries (i.e.,
/// [`InferenceError`]).
#[allow(missing_docs)]
#[derive(Debug)]
pub enum SetupError {
    Inference(InferenceError),
    Loading(LoadingError),
}

impl Error for SetupError {}

impl fmt::Display for SetupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Inference(error) => write!(f, "inference error: {error}"),
            Self::Loading(error) => write!(f, "library loading error: {error}"),
        }
    }
}

impl From<InferenceError> for SetupError {
    fn from(error: InferenceError) -> Self {
        SetupError::Inference(error)
    }
}

impl From<LoadingError> for SetupError {
    fn from(error: LoadingError) -> Self {
        SetupError::Loading(error)
    }
}

/// Returns the last error message from the OpenVINO library.
///
/// Note: With the current API, error messages are automatically captured and included
/// in `InferenceError` instances. This function is provided for direct C API access
/// if needed, but typically you should rely on the message in the error itself.
pub fn get_last_error_message() -> Option<String> {
    unsafe {
        let ptr = openvino_sys::ov_get_last_err_msg();
        if ptr.is_null() {
            None
        } else {
            Some(CStr::from_ptr(ptr).to_string_lossy().into_owned())
        }
    }
}
