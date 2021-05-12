//! Problem definition.

use crate::cost::CostFn;

pub trait Problem {
    fn t_end(&self) -> i32;
    fn inc_t_end(&mut self);
}

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
impl<'a, T> Problem for SmoothedConvexOptimization<'a, T> {
    fn t_end(&self) -> i32 {
        self.t_end
    }
    fn inc_t_end(&mut self) {
        self.t_end += 1
    }
}

/// Smoothed Load Optimization
pub struct SmoothedLoadOptimization<T> {
    /// Number of dimensions.
    pub d: i32,
    /// Finite, positive time horizon.
    pub t_end: i32,
    /// Vector of upper bounds of each dimension.
    pub bounds: Vec<T>,
    /// Vector of positive real constants resembling the switching cost of each dimension (strictly descending).
    /// Dimensions must be _efficient_, i.e. there must not be dimensions with a higher switching and higher hitting cost than onether dimension.
    pub switching_cost: Vec<f64>,
    /// Time-independent cost of each dimension (strictly ascending).
    pub hitting_cost: Vec<f64>,
    /// Load at each time step `t`.
    pub load: Vec<i32>,
}
pub type DiscreteSmoothedLoadOptimization = SmoothedLoadOptimization<i32>;
impl<T> Problem for SmoothedLoadOptimization<T> {
    fn t_end(&self) -> i32 {
        self.t_end
    }
    fn inc_t_end(&mut self) {
        self.t_end += 1
    }
}
