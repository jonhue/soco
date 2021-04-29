//! Type definitions.

use std::sync::Arc;

/// Cost function. Must be total on 0<=j<=m. May return `None` otherwise.
pub type CostFn<'a, T> = Arc<dyn Fn(i32, T) -> Option<f64> + 'a>;

/// Data-Center Right-Sizing problem.
pub enum Problem<'a, T> {
    Hom(HomProblem<'a, T>),
}

/// Homogeneous Data-Center Right-Sizing problem.
pub struct HomProblem<'a, T> {
    /// Number of servers.
    pub m: i32,
    /// Finite, positive time horizon.
    pub t_end: i32,
    /// Positive real constant resembling the switching cost.
    pub beta: f64,
    /// Non-negative convex cost functions.
    /// Must be total on 1<=t<=T, 0<=j<=m. May return `None` otherwise.
    pub f: CostFn<'a, T>,
}
pub type DiscreteHomProblem<'a> = HomProblem<'a, i32>;
pub type ContinuousHomProblem<'a> = HomProblem<'a, f64>;

/// Online instance of a problem.
pub struct Online<T> {
    /// Problem.
    pub p: T,
    /// Finite, non-negative prediction window.
    pub w: i32,
}

/// Solution fragment at some time t to an online problem.
///
/// * `T` - Number of servers at time t.
/// * `U` - Memory.
pub type OnlineSolution<T, U> = (T, U);

/// Result of the Homogeneous Data-Center Right-Sizing problem.
/// Number of active servers from time 1 to time T.
pub type Schedule<T> = Vec<T>;
pub type DiscreteSchedule = Schedule<i32>;
pub type ContinuousSchedule = Schedule<f64>;
