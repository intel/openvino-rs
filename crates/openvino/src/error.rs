use openvino_sys::ov_status_e;
use std::error::Error;
use std::fmt;

/// See
/// [`ov_status_e`](https://docs.openvino.ai/2024/api/c_cpp_api/group__ov__base__c__api.html#_CPPv411ov_status_e);
/// enumerates errors returned by the OpenVINO implementation.
#[allow(missing_docs)]
#[derive(Debug, PartialEq, Eq)]
pub enum InferenceError {
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

impl InferenceError {
    /// Convert an `openvino_sys` error to a [`Result`]:
    /// - `0` becomes `Ok`
    /// - anything else becomes `Err` containing an [`InferenceError`]
    pub fn convert(status: ov_status_e) -> Result<(), InferenceError> {
        match status {
            ov_status_e::OK => Ok(()),
            ov_status_e::GENERAL_ERROR => Err(Self::GeneralError),
            ov_status_e::NOT_IMPLEMENTED => Err(Self::NotImplemented),
            ov_status_e::NETWORK_NOT_LOADED => Err(Self::NetworkNotLoaded),
            ov_status_e::PARAMETER_MISMATCH => Err(Self::ParameterMismatch),
            ov_status_e::NOT_FOUND => Err(Self::NotFound),
            ov_status_e::OUT_OF_BOUNDS => Err(Self::OutOfBounds),
            ov_status_e::UNEXPECTED => Err(Self::Unexpected),
            ov_status_e::REQUEST_BUSY => Err(Self::RequestBusy),
            ov_status_e::RESULT_NOT_READY => Err(Self::ResultNotReady),
            ov_status_e::NOT_ALLOCATED => Err(Self::NotAllocated),
            ov_status_e::INFER_NOT_STARTED => Err(Self::InferNotStarted),
            ov_status_e::NETWORK_NOT_READ => Err(Self::NetworkNotRead),
            ov_status_e::INFER_CANCELLED => Err(Self::InferCancelled),
            ov_status_e::INVALID_C_PARAM => Err(Self::InvalidCParam),
            ov_status_e::UNKNOWN_C_ERROR => Err(Self::UnknownCError),
            ov_status_e::NOT_IMPLEMENT_C_METHOD => Err(Self::NotImplementCMethod),
            ov_status_e::UNKNOW_EXCEPTION => Err(Self::UnknownException),
        }
    }
}

impl Error for InferenceError {}

impl fmt::Display for InferenceError {
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
