//! Definition of schedules.

/// Result of the Homogeneous Data-Center Right-Sizing problem.
/// Number of active servers from time 1 to time T.
pub type Schedule<T> = Vec<T>;
pub type DiscreteSchedule = Schedule<i32>;
pub type ContinuousSchedule = Schedule<f64>;
