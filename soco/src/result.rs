//! Result types.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Failure {
    #[error("A verifier determined an invalidity: {0}")]
    Invalid(String),
    #[error("The given matrix must be invertible to compute the Mahalanobis distance.")]
    MatrixMustBeInvertible,
    #[error("When solving an online problem from a given time slot, the property `t_end` (current time slot) must always be one time slot ahead of the length of the obtained schedule (number of previous time slots). Yet, the number of previous time slots is {previous_time_slots} and the current time slot is {current_time_slot}.")]
    OnlineInconsistentCurrentTimeSlot {
        previous_time_slots: i32,
        current_time_slot: i32,
    },
    #[error("When solving an online problem from a given time slot, the accumulated memory up to this time slot must be provided. Yet, the number of previous time slots is {previous_time_slots} and the memory consists of {memory_entries} entries.")]
    OnlineOutOfDateMemory {
        previous_time_slots: i32,
        memory_entries: i32,
    },
    #[error("This algorithm does not support inverted movement costs. Set `inverted = false`.")]
    UnsupportedInvertedCost,
    #[error("This algorithm does not support `L`-constrained movement. Set `l = None`.")]
    UnsupportedLConstrainedMovement,
    #[error("This online algorithm does not support a prediction window. Set `w = 0` (was {0}).")]
    UnsupportedPredictionWindow(i32),
    #[error("This online algorithm does not support multi-dimensional problems. Set `d = 1` (was {0}).")]
    UnsupportedProblemDimension(i32),
}

pub type Result<T> = std::result::Result<T, Failure>;
