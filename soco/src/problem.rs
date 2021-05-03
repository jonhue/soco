//! Problem definition.

use crate::cost::CostFn;

/// Multi-Dimensional Smoothed Convex Optimization.
pub struct Problem<'a, T> {
    /// Number of dimensions.
    pub d: i32,
    /// Finite, positive time horizon.
    pub t_end: i32,
    /// Vector of upper bounds of each dimension.
    pub bounds: Vec<T>,
    /// Vector of positive real constants resembling the switching cost of each dimension.
    pub switching_costs: Vec<f64>,
    /// Non-negative convex cost functions.
    pub f: CostFn<'a, Vec<T>>,
}
pub type DiscreteProblem<'a> = Problem<'a, i32>;
pub type ContinuousProblem<'a> = Problem<'a, f64>;
