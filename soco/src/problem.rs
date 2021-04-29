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
