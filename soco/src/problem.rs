//! Problem definition.

use crate::config::Config;
use crate::cost::CallableCostFn;
use crate::cost::CostFn;
use crate::norm::NormFn;
use crate::value::Value;

/// Trait implemented by all finite-time-horizon problems.
pub trait Problem: Clone {
    /// Number of dimensions.
    fn d(&self) -> i32;
    /// Finite, positive time horizon.
    fn t_end(&self) -> i32;
    /// Increases the time horizon by one time step.
    fn inc_t_end(&mut self);
}

macro_rules! impl_problem {
    ($T:ty) => {
        impl<'a, T> Problem for $T
        where
            T: Value,
        {
            fn d(&self) -> i32 {
                self.d
            }
            fn t_end(&self) -> i32 {
                self.t_end
            }
            fn inc_t_end(&mut self) {
                self.t_end += 1
            }
        }
    };
}

/// Gives type a default value which may depend on a problem instance.
pub trait DefaultGivenProblem<P>
where
    P: Problem,
{
    fn default(p: &P) -> Self;
}
impl<T, P> DefaultGivenProblem<P> for T
where
    T: Default,
    P: Problem,
{
    fn default(_: &P) -> Self {
        T::default()
    }
}

/// Online instance of a problem.
#[derive(Clone)]
pub struct Online<T> {
    /// Problem.
    pub p: T,
    /// Finite, non-negative prediction window.
    ///
    /// This prediction window is included in the time bound of the problem instance,
    /// i.e. at time `t` `t_end` should be set to `t + w`.
    pub w: i32,
}

/// Smoothed Convex Optimization.
#[derive(Clone)]
pub struct SmoothedConvexOptimization<'a, T>
where
    T: Value + 'a,
{
    /// Number of dimensions.
    pub d: i32,
    /// Finite, positive time horizon.
    pub t_end: i32,
    /// Vector of lower and upper bounds of each dimension.
    pub bounds: Vec<(T, T)>,
    /// Norm function.
    pub switching_cost: NormFn<'a, Config<T>>,
    /// Non-negative convex cost functions.
    pub hitting_cost: CostFn<'a, Config<T>>,
}
impl_problem!(SmoothedConvexOptimization<'a, T>);
impl<'a, T> SmoothedConvexOptimization<'a, T>
where
    T: Value,
{
    pub fn hit_cost(&self, t: i32, x: Config<T>) -> f64 {
        self.hitting_cost.call(t, x, &self.bounds)
    }
}
pub type FractionalSmoothedConvexOptimization<'a> =
    SmoothedConvexOptimization<'a, f64>;

/// Simplified Smoothed Convex Optimization.
#[derive(Clone)]
pub struct SimplifiedSmoothedConvexOptimization<'a, T>
where
    T: Value + 'a,
{
    /// Number of dimensions.
    pub d: i32,
    /// Finite, positive time horizon.
    pub t_end: i32,
    /// Vector of upper bounds of each dimension.
    pub bounds: Vec<T>,
    /// Vector of positive real constants resembling the switching cost of each dimension.
    pub switching_cost: Vec<f64>,
    /// Non-negative convex cost functions.
    pub hitting_cost: CostFn<'a, Config<T>>,
}
impl_problem!(SimplifiedSmoothedConvexOptimization<'a, T>);
impl<'a, T> SimplifiedSmoothedConvexOptimization<'a, T>
where
    T: Value,
{
    pub fn hit_cost(&self, t: i32, x: Config<T>) -> f64 {
        self.hitting_cost.call(t, x, &self.bounds)
    }
}
pub type IntegralSimplifiedSmoothedConvexOptimization<'a> =
    SimplifiedSmoothedConvexOptimization<'a, i32>;
pub type FractionalSimplifiedSmoothedConvexOptimization<'a> =
    SimplifiedSmoothedConvexOptimization<'a, f64>;

/// Smoothed Balanced-Load Optimization.
#[derive(Clone)]
pub struct SmoothedBalancedLoadOptimization<'a, T>
where
    T: Value,
{
    /// Number of dimensions.
    pub d: i32,
    /// Finite, positive time horizon.
    pub t_end: i32,
    /// Vector of upper bounds of each dimension.
    pub bounds: Vec<T>,
    /// Vector of positive real constants resembling the switching cost of each dimension.
    pub switching_cost: Vec<f64>,
    /// Positive increasing cost functions for each dimension.
    pub hitting_cost: Vec<CostFn<'a, T>>,
    /// Non-negative load at each time step `t`.
    pub load: Vec<T>,
}
impl_problem!(SmoothedBalancedLoadOptimization<'a, T>);
impl<'a, T> SmoothedBalancedLoadOptimization<'a, T>
where
    T: Value,
{
    pub fn hit_cost(&self, t: i32, x: Config<T>) -> f64 {
        assert!(self.d == x.d());
        let mut result = 0.;
        for k in 0..self.d as usize {
            result += self.hitting_cost[k].call(t, x[k], self.bounds[k]);
        }
        result
    }
}
pub type IntegralSmoothedBalancedLoadOptimization<'a> =
    SmoothedBalancedLoadOptimization<'a, i32>;

/// Smoothed Load Optimization.
#[derive(Clone)]
pub struct SmoothedLoadOptimization<T>
where
    T: Value,
{
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
    /// Non-negative load at each time step `t`.
    pub load: Vec<T>,
}
impl_problem!(SmoothedLoadOptimization<T>);
pub type IntegralSmoothedLoadOptimization = SmoothedLoadOptimization<i32>;
