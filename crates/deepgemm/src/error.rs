use std::{error::Error as StdError, ffi::CStr, fmt};

/// Result type used by the DeepGEMM safe wrapper layer.
pub type Result<T> = std::result::Result<T, Error>;

/// Error returned by the DeepGEMM safe wrapper layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// A caller-provided argument was invalid.
    InvalidArgument(String),
    /// The selected GPU architecture is not supported.
    UnsupportedArch(String),
    /// CUDA returned an error.
    Cuda(String),
    /// DeepGEMM hit an unexpected internal error.
    Internal(String),
    /// The raw C ABI returned an unknown status code.
    UnknownStatus {
        /// Raw status code returned by the C ABI.
        status: deepgemm_sys::deepgemm_status_t,
        /// Error message reported by the C ABI.
        message: String,
    },
}

impl Error {
    /// Converts a raw DeepGEMM C ABI status and message into a safe error.
    pub fn from_raw_status(
        status: deepgemm_sys::deepgemm_status_t,
        message: impl Into<String>,
    ) -> Self {
        let message = message.into();
        match status {
            deepgemm_sys::DEEPGEMM_STATUS_INVALID_ARGUMENT => Self::InvalidArgument(message),
            deepgemm_sys::DEEPGEMM_STATUS_UNSUPPORTED_ARCH => Self::UnsupportedArch(message),
            deepgemm_sys::DEEPGEMM_STATUS_CUDA_ERROR => Self::Cuda(message),
            deepgemm_sys::DEEPGEMM_STATUS_INTERNAL_ERROR => Self::Internal(message),
            _ => Self::UnknownStatus { status, message },
        }
    }

    /// Returns the last C ABI error as a Rust string.
    pub fn last_raw_error() -> String {
        // SAFETY: `deepgemm_last_error` returns a process-local NUL-terminated string owned by the
        // native shim. A null pointer is treated as an empty message.
        let ptr = unsafe { deepgemm_sys::deepgemm_last_error() };
        if ptr.is_null() {
            return String::new();
        }
        // SAFETY: the native shim guarantees a valid NUL-terminated string for non-null pointers.
        unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned()
    }

    /// Converts a C ABI status into `Result<()>`.
    pub fn check_raw_status(status: deepgemm_sys::deepgemm_status_t) -> Result<()> {
        if status == deepgemm_sys::DEEPGEMM_STATUS_SUCCESS {
            return Ok(());
        }
        Err(Self::from_raw_status(status, Self::last_raw_error()))
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidArgument(message) => write!(formatter, "invalid argument: {message}"),
            Self::UnsupportedArch(message) => {
                write!(formatter, "unsupported architecture: {message}")
            }
            Self::Cuda(message) => write!(formatter, "CUDA error: {message}"),
            Self::Internal(message) => write!(formatter, "internal DeepGEMM error: {message}"),
            Self::UnknownStatus { status, message } => {
                write!(formatter, "unknown DeepGEMM status {status}: {message}")
            }
        }
    }
}

impl StdError for Error {}
