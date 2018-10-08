//! Customs error management for the entire crate.

use std::error::Error as StdError;
use std::result;
use std::{fmt, io};

use bincode::Error as BincodeError;

/// A type alias for `Result<T, GenError>`.
pub type GenResult<T> = result::Result<T, GenError>;


/// An error that can occur when interacting with the algorithm.
#[derive(Debug)]
pub struct GenError(Box<ErrorKind>);

/// A crate private constructor for `Error`.
pub(crate) fn new_gen_error(kind: ErrorKind) -> GenError {
    GenError(Box::new(kind))
}

impl GenError {
    /// Return the specific type of this error.
    pub fn kind(&self) -> &ErrorKind {
        &self.0
    }


    /// Unwrap this error into its underlying type.
    pub fn into_kind(self) -> ErrorKind {
        *self.0
    }
}


/// The specific type of an error.
#[derive(Debug)]
pub enum ErrorKind {
    /// An I/O error that occurred while processing a data stream.
    Io(io::Error),
    /// An error of this kind occurs only when using the Bincode (ser)deserializer.
    SerDeserializeError(BincodeError),
    /// Hints that destructuring should not be exhaustive.
    ///
    /// This enum may grow additional variants, so this makes sure clients
    /// don't count on exhaustive matching. (Otherwise, adding a new variant
    /// could break existing code.)
    #[doc(hidden)]
    __Nonexhaustive,
}


impl StdError for GenError {
    fn description(&self) -> &str {
        match *self.0 {
            ErrorKind::Io(ref err) => err.description(),
            ErrorKind::SerDeserializeError(ref err) => err.description(),
            _ => unreachable!(),
        }
    }


    fn cause(&self) -> Option<&StdError> {
        match *self.0 {
            ErrorKind::Io(ref err) => Some(err),
            ErrorKind::SerDeserializeError(ref err) => Some(err),
            _ => unreachable!(),
        }
    }
}


impl fmt::Display for GenError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self.0 {
            ErrorKind::Io(ref err) => err.fmt(f),
            ErrorKind::SerDeserializeError(ref err) => err.fmt(f),
            _ => unreachable!(),
        }
    }
}


impl From<io::Error> for GenError {
    fn from(err: io::Error) -> Self {
        new_gen_error(ErrorKind::Io(err))
    }
}


impl From<GenError> for io::Error {
    fn from(err: GenError) -> Self {
        io::Error::new(io::ErrorKind::Other, err)
    }
}


impl From<BincodeError> for GenError {
    fn from(err: BincodeError) -> Self {
        new_gen_error(ErrorKind::SerDeserializeError(err))
    }
}
