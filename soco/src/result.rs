//! Result types.

#[derive(Debug)]
pub enum Error {
    Bisection(String),
    BoundMustBeGreaterThanZero,
    CostFnMustBeTotal,
    DimensionInconsistent,
    Integration(String),
    Invalid(String),
    InvalidIntegrationInterval,
    LcpBoundComputationExceedsDomain,
    LcpBoundMismatch(f64, f64),
    MatrixMustBeInvertible,
    MemoryShouldBePresent,
    MinimizerShouldBeBounded,
    MustBePowOf2,
    NlOpt(nlopt::FailState),
    OnlineInsufficientInformation(String),
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
