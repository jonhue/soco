//! Problem definition.

use crate::config::Config;
use crate::cost::CostFn;
use crate::model::data_center::loads::{
    apply_loads_over_time, LoadFractions, LoadProfile,
};
use crate::model::data_center::safe_balancing;
use crate::norm::NormFn;
use crate::value::Value;
use crate::verifiers::VerifiableProblem;
use noisy_float::prelude::*;
use num::NumCast;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

/// Trait implemented by all finite-time-horizon problems.
pub trait Problem: Clone + std::fmt::Debug + Send + VerifiableProblem {
    /// Number of dimensions.
    fn d(&self) -> i32;
    /// Finite, positive time horizon.
    fn t_end(&self) -> i32;
    /// Updates the time horizon.
    fn set_t_end(&mut self, t_end: i32);
    /// Increases the time horizon by one time step.
    fn inc_t_end(&mut self) {
        self.set_t_end(self.t_end() + 1)
    }
}

macro_rules! impl_problem {
    ($T:ty) => {
        impl<'a, T> Problem for $T
        where
            T: Value<'a>,
        {
            fn d(&self) -> i32 {
                self.d
            }
            fn t_end(&self) -> i32 {
                self.t_end
            }
            fn set_t_end(&mut self, t_end: i32) {
                self.t_end = t_end
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
#[derive(Clone, Debug)]
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
#[derive(Clone, Derivative)]
#[derivative(Debug)]
pub struct SmoothedConvexOptimization<'a, T> {
    /// Number of dimensions.
    pub d: i32,
    /// Finite, positive time horizon.
    pub t_end: i32,
    /// Vector of lower and upper bounds of each dimension.
    pub bounds: Vec<(T, T)>,
    /// Norm function.
    #[derivative(Debug = "ignore")]
    pub switching_cost: NormFn<'a, T>,
    /// Non-negative convex cost functions.
    #[derivative(Debug = "ignore")]
    pub hitting_cost: CostFn<'a, Config<T>>,
}
impl_problem!(SmoothedConvexOptimization<'a, T>);
impl<'a, T> SmoothedConvexOptimization<'a, T>
where
    T: Value<'a>,
{
    pub fn hit_cost(&self, t: i32, x: Config<T>) -> N64 {
        self.hitting_cost.call(t, x, &self.bounds)
    }
}
pub type FractionalSmoothedConvexOptimization<'a> =
    SmoothedConvexOptimization<'a, f64>;

/// Simplified Smoothed Convex Optimization.
#[derive(Clone, Derivative)]
#[derivative(Debug)]
pub struct SimplifiedSmoothedConvexOptimization<'a, T> {
    /// Number of dimensions.
    pub d: i32,
    /// Finite, positive time horizon.
    pub t_end: i32,
    /// Vector of upper bounds of each dimension.
    pub bounds: Vec<T>,
    /// Vector of positive real constants resembling the switching cost of each dimension.
    pub switching_cost: Vec<f64>,
    /// Non-negative convex cost functions.
    #[derivative(Debug = "ignore")]
    pub hitting_cost: CostFn<'a, Config<T>>,
}
impl_problem!(SimplifiedSmoothedConvexOptimization<'a, T>);
impl<'a, T> SimplifiedSmoothedConvexOptimization<'a, T>
where
    T: Value<'a>,
{
    pub fn hit_cost(&self, t: i32, x: Config<T>) -> N64 {
        self.hitting_cost.call(t, x, &self.bounds)
    }
}
pub type IntegralSimplifiedSmoothedConvexOptimization<'a> =
    SimplifiedSmoothedConvexOptimization<'a, i32>;
pub type FractionalSimplifiedSmoothedConvexOptimization<'a> =
    SimplifiedSmoothedConvexOptimization<'a, f64>;

/// Smoothed Balanced-Load Optimization.
#[derive(Clone, Derivative)]
#[derivative(Debug)]
pub struct SmoothedBalancedLoadOptimization<'a, T> {
    /// Number of dimensions.
    pub d: i32,
    /// Finite, positive time horizon.
    pub t_end: i32,
    /// Vector of upper bounds of each dimension.
    pub bounds: Vec<T>,
    /// Vector of positive real constants resembling the switching cost of each dimension.
    pub switching_cost: Vec<f64>,
    /// Positive increasing cost functions for each dimension.
    #[derivative(Debug = "ignore")]
    pub hitting_cost: Vec<CostFn<'a, f64>>,
    /// Non-negative load at each time step.
    pub load: Vec<T>,
}
impl_problem!(SmoothedBalancedLoadOptimization<'a, T>);
impl<'a, T> SmoothedBalancedLoadOptimization<'a, T>
where
    T: Value<'a>,
{
    pub fn hit_cost(self, t: i32, x: Config<T>) -> N64 {
        let bounds = self.bounds.clone();
        let loads = self
            .load
            .par_iter()
            .map(|&l| LoadProfile::single(NumCast::from(l).unwrap()))
            .collect();
        apply_loads_over_time(
            self.d,
            1,
            move |t: i32,
                  x_: &Config<T>,
                  lambda: &LoadProfile,
                  zs: &LoadFractions| {
                assert!(self.d == x_.d());
                (0..self.d as usize)
                    .map(|k| -> N64 {
                        let total_load = zs.select_loads(lambda, k)[0];
                        let x = NumCast::from(x_[k]).unwrap();
                        safe_balancing(x, total_load, || {
                            x * self.hitting_cost[k].call(
                                t,
                                (total_load / x).raw(),
                                &self.bounds[k],
                            )
                        })
                    })
                    .sum::<N64>()
            },
            loads,
            1,
        )
        .call(t, x, &bounds)
    }
}
pub type IntegralSmoothedBalancedLoadOptimization<'a> =
    SmoothedBalancedLoadOptimization<'a, i32>;

/// Smoothed Load Optimization.
#[derive(Clone, Debug)]
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
    /// Non-negative load at each time step.
    pub load: Vec<T>,
}
impl_problem!(SmoothedLoadOptimization<T>);
pub type IntegralSmoothedLoadOptimization = SmoothedLoadOptimization<i32>;
