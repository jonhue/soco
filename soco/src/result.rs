//! Result types.

#[derive(Debug)]
pub enum Error {
    CostFnMustBeTotal,
    NLOpt(nlopt::FailState),
    Other(String),
}

impl From<String> for Error {
    fn from(error: String) -> Self {
        Error::Other(error)
    }
}

impl From<nlopt::FailState> for Error {
    fn from(error: nlopt::FailState) -> Self {
        Error::NLOpt(error)
    }
}

impl From<(nlopt::FailState, f64)> for Error {
    fn from(error: (nlopt::FailState, f64)) -> Self {
        Error::NLOpt(error.0)
    }
}

pub type Result<T> = std::result::Result<T, Error>;
