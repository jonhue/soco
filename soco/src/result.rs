//! Result types.

#[derive(Debug)]
pub enum Error {
    CostFnMustBeTotal,
    Invalid(String),
    MustBePowOf2,
    NlOpt(nlopt::FailState),
    OnlineInsufficientInformation,
    Other(String),
    UnsupportedPredictionWindow,
}

impl From<String> for Error {
    fn from(error: String) -> Self {
        Error::Other(error)
    }
}

impl From<nlopt::FailState> for Error {
    fn from(error: nlopt::FailState) -> Self {
        Error::NlOpt(error)
    }
}

impl From<(nlopt::FailState, f64)> for Error {
    fn from(error: (nlopt::FailState, f64)) -> Self {
        Error::NlOpt(error.0)
    }
}

pub type Result<T> = std::result::Result<T, Error>;
