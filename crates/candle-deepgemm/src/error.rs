use std::fmt;

/// Result type used by the Candle DeepGEMM integration crate.
pub type Result<T> = std::result::Result<T, Error>;

/// Error returned by Candle DeepGEMM integration APIs.
#[derive(Debug)]
pub enum Error {
    /// Error returned by the Candle-independent DeepGEMM wrapper.
    DeepGemm(deepgemm::Error),
    /// Error returned by Candle.
    Candle(candle::Error),
    /// Tensor validation or pointer extraction failed.
    Tensor(String),
}

impl From<deepgemm::Error> for Error {
    fn from(error: deepgemm::Error) -> Self {
        Self::DeepGemm(error)
    }
}

impl From<candle::Error> for Error {
    fn from(error: candle::Error) -> Self {
        Self::Candle(error)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::DeepGemm(error) => write!(f, "{error}"),
            Error::Candle(error) => write!(f, "{error}"),
            Error::Tensor(message) => write!(f, "tensor error: {message}"),
        }
    }
}

impl std::error::Error for Error {}

pub(crate) fn invalid_arg<T>(message: impl Into<String>) -> Result<T> {
    Err(Error::Tensor(message.into()))
}
