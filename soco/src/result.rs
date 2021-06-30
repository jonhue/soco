//! Result types.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum Failure {
    #[error("Bisecting failed with message: {0}")]
    Bisection(String),
    // BoundMustBeGreaterThanZero,
    // DimensionInconsistent,
    #[error("Integrating failed with message: {0}")]
    Integration(String),
    #[error("A verifier determined an invalidity: {0}")]
    Invalid(String),
    #[error("The interval from {from} to {to} is invalid.")]
    InvalidInterval { from: f64, to: f64 },
    // LcpBoundComputationExceedsDomain,
    // LcpBoundMismatch(f64, f64),
    #[error("The given matrix must be invertible to compute the Mahalanobis distance.")]
    MatrixMustBeInvertible,
    // MemoryShouldBePresent,
    // MinimizerShouldBeBounded,
    #[error("The uni-dimensional optimal integral algorithm requires that the bound `m` is a power of two. Use `make_pow_of_2` to transform your problem instance before use.")]
    MustBePowOf2,
    #[error("NLopt returned with an error.")]
    NlOpt(nlopt::FailState),
    #[error("When solving an online problem, the time horizon `T` should equal the current time slot plus the prediction window. But instead we have `T = {t_end}`, `t = {t}`, and `w = {w}`.")]
    OnlineInsufficientInformation { t_end: i32, t: i32, w: i32 },
    #[error("When solving an online problem from a given time slot, the accumulated memory up to this time slot must be provided. Yet, the number of previous time slots is {previous_time_slots} and the memory consists of {memory_entries} entries.")]
    OnlineOutOfDateMemory {
        previous_time_slots: i32,
        memory_entries: i32,
    },
    // PathsShouldBeCached,
    #[error("This algorithm expects a problem instance which is the relaxed problem of an integral problem. In particular, the bounds shouldn't be fractional.")]
    MustBeRelaxedProblem,
    // SubpathShouldBePresent,
    // UnsupportedArgument(String),
    // UnsupportedBoundsCalculation,
    #[error("This online algorithm does not support a prediction window. Set `w = 0` (was {0}).")]
    UnsupportedPredictionWindow(i32),
    #[error("This online algorithm does not support multi-dimensional problems. Set `d = 1` (was {0}).")]
    UnsupportedProblemDimension(i32),
}

impl From<nlopt::FailState> for Failure {
    fn from(error: nlopt::FailState) -> Self {
        Failure::NlOpt(error)
    }
}

impl From<(nlopt::FailState, f64)> for Failure {
    fn from(error: (nlopt::FailState, f64)) -> Self {
        Failure::NlOpt(error.0)
    }
}

pub type Result<T> = std::result::Result<T, Failure>;
