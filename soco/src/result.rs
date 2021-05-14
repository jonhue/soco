//! Result types.

#[derive(Debug)]
pub enum Error {
    CostFnMustBeTotal,
    GammaTooSmall,
    Integration(String),
    Invalid(String),
    LcpBoundComputationExceedsDomain,
    LcpBoundMismatch(f64, f64),
    MustBePowOf2,
    NlOpt(nlopt::FailState),
    OnlineInsufficientInformation,
    PathsShouldBeCached,
    SubpathShouldBePresent,
    UnsupportedArgument(String),
    UnsupportedBoundsCalculation,
    UnsupportedPredictionWindow,
    UnsupportedProblemDimension,
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
