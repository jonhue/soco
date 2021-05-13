//! Definition of schedules.

/// For some time `t`, assigns each dimension `d` a unique value.
pub type Config<T> = Vec<T>;

/// Includes all steps from time `1` to time `t_end`.
pub type Schedule<T> = Vec<Config<T>>;
pub type DiscreteSchedule = Schedule<i32>;
pub type ContinuousSchedule = Schedule<f64>;
