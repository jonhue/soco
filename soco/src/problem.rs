//! Problem definition.

use crate::cost::CostFn;

/// Smoothed Convex Optimization.
pub struct SmoothedConvexOptimization<'a, T> {
    /// Number of dimensions.
    pub d: i32,
    /// Finite, positive time horizon.
    pub t_end: i32,
    /// Vector of upper bounds of each dimension.
    pub bounds: Vec<T>,
    /// Vector of positive real constants resembling the switching cost of each dimension.
    pub switching_cost: Vec<f64>,
    /// Non-negative convex cost functions.
    pub hitting_cost: CostFn<'a, Vec<T>>,
}
pub type DiscreteSmoothedConvexOptimization<'a> =
    SmoothedConvexOptimization<'a, i32>;
pub type ContinuousSmoothedConvexOptimization<'a> =
    SmoothedConvexOptimization<'a, f64>;

/// Smoothed Constant Optimization
pub struct SmoothedConstantOptimization<T> {
    /// Number of dimensions.
    pub d: i32,
    /// Finite, positive time horizon.
    pub t_end: i32,
    /// Vector of upper bounds of each dimension.
    pub bounds: Vec<T>,
    /// Vector of positive real constants resembling the switching cost of each dimension.
    /// Dimensions must be _efficient_, i.e. there must not be dimensions with a higher switching and higher hitting cost than onether dimension.
    pub switching_cost: Vec<f64>,
    /// Time-independent cost of each dimension (ascending).
    pub hitting_cost: Vec<f64>,
}
pub type DiscreteSmoothedConstantOptimization =
    SmoothedConstantOptimization<i32>;
