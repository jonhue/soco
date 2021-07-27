//! Problem definition.

use crate::config::Config;
use crate::cost::{Cost, CostFn, FailableCost, FailableCostFn};
use crate::model::data_center::loads::{
    apply_loads_over_time, LoadFractions, LoadProfile,
};
use crate::model::data_center::model::{
    DataCenterModelOutputFailure, DataCenterModelOutputSuccess,
    DataCenterObjective,
};
use crate::model::data_center::safe_balancing;
use crate::model::{ModelOutput, ModelOutputFailure, ModelOutputSuccess};
use crate::norm::NormFn;
use crate::objective::scalar_movement;
use crate::value::Value;
use crate::verifiers::VerifiableProblem;
use noisy_float::prelude::*;
use num::NumCast;

/// Trait implemented by all finite-time-horizon problems.
pub trait BaseProblem:
    Clone + std::fmt::Debug + Send + VerifiableProblem
{
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
macro_rules! impl_base_problem {
    ($T:ty, $C:tt, $D:tt) => {
        impl<'a, T, $C, $D> BaseProblem for $T
        where
            T: Value<'a>,
            C: Clone,
            D: Clone,
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
    ($T:ty) => {
        impl<'a, T> BaseProblem for $T
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

pub trait Problem<T, C, D>: BaseProblem
where
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    fn hit_cost(&self, t: i32, x: Config<T>) -> Cost<C, D>;

    fn raw_hit_cost(&self, t: i32, x: Config<T>) -> N64 {
        self.hit_cost(t, x).cost
    }

    fn movement(&self, prev_x: Config<T>, x: Config<T>, inverted: bool) -> N64;
}

/// Gives type a default value which may depend on a problem instance.
pub trait DefaultGivenProblem<T, P, C, D>
where
    P: Problem<T, C, D>,
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    fn default(p: &P) -> Self;
}
impl<T, P, C, D, U> DefaultGivenProblem<T, P, C, D> for U
where
    U: Default,
    P: Problem<T, C, D>,
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    fn default(_: &P) -> Self {
        U::default()
    }
}

/// Online instance of a problem.
#[derive(Clone, Debug)]
pub struct Online<P> {
    /// Problem.
    pub p: P,
    /// Finite, non-negative prediction window.
    ///
    /// This prediction window is included in the time bound of the problem instance,
    /// i.e. at time `t` `t_end` should be set to `t + w`.
    pub w: i32,
}

/// Smoothed Convex Optimization.
#[derive(Clone, Derivative)]
#[derivative(Debug)]
pub struct SmoothedConvexOptimization<'a, T, C, D> {
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
    pub hitting_cost: CostFn<'a, Config<T>, C, D>,
}
impl_base_problem!(SmoothedConvexOptimization<'a, T, C, D>, C, D);
impl<'a, T, C, D> Problem<T, C, D> for SmoothedConvexOptimization<'a, T, C, D>
where
    T: Value<'a>,
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    fn hit_cost(&self, t: i32, x: Config<T>) -> Cost<C, D> {
        self.hitting_cost
            .call_certain_within_bounds(t, x, &self.bounds)
    }

    fn movement(&self, prev_x: Config<T>, x: Config<T>, inverted: bool) -> N64 {
        assert!(!inverted, "Unsupported inverted movement.");

        (self.switching_cost)(x - prev_x)
    }
}
pub type FractionalSmoothedConvexOptimization<'a, C, D> =
    SmoothedConvexOptimization<'a, f64, C, D>;

/// Simplified Smoothed Convex Optimization.
#[derive(Clone, Derivative)]
#[derivative(Debug)]
pub struct SimplifiedSmoothedConvexOptimization<'a, T, C, D> {
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
    pub hitting_cost: CostFn<'a, Config<T>, C, D>,
}
impl_base_problem!(SimplifiedSmoothedConvexOptimization<'a, T, C, D>, C, D);
impl<'a, T, C, D> Problem<T, C, D>
    for SimplifiedSmoothedConvexOptimization<'a, T, C, D>
where
    T: Value<'a>,
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    fn hit_cost(&self, t: i32, x: Config<T>) -> Cost<C, D> {
        self.hitting_cost
            .call_certain_within_bounds(t, x, &self.bounds)
    }

    fn movement(&self, prev_x: Config<T>, x: Config<T>, inverted: bool) -> N64 {
        movement(self.d, &self.switching_cost, prev_x, x, inverted)
    }
}
pub type IntegralSimplifiedSmoothedConvexOptimization<'a, C, D> =
    SimplifiedSmoothedConvexOptimization<'a, i32, C, D>;
pub type FractionalSimplifiedSmoothedConvexOptimization<'a, C, D> =
    SimplifiedSmoothedConvexOptimization<'a, f64, C, D>;

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
    pub hitting_cost:
        Vec<FailableCostFn<'a, f64, DataCenterModelOutputFailure>>,
    /// Non-negative load at each time step.
    pub load: Vec<T>,
}
impl_base_problem!(SmoothedBalancedLoadOptimization<'a, T>);
impl<'a, T>
    Problem<T, DataCenterModelOutputSuccess, DataCenterModelOutputFailure>
    for SmoothedBalancedLoadOptimization<'a, T>
where
    T: Value<'a>,
{
    fn hit_cost(
        &self,
        t: i32,
        x: Config<T>,
    ) -> Cost<DataCenterModelOutputSuccess, DataCenterModelOutputFailure> {
        let bounds = self.bounds.clone();
        let loads = self
            .load
            .iter()
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
                DataCenterObjective {
                    energy_cost: (0..self.d as usize)
                        .map(|k| -> N64 {
                            let total_load = zs.select_loads(lambda, k)[0];
                            let x = NumCast::from(x_[k]).unwrap();
                            safe_balancing(x, total_load, || {
                                x * self.hitting_cost[k]
                                    .call_certain(t, (total_load / x).raw())
                                    .cost
                            })
                        })
                        .sum::<N64>(),
                    revenue_loss: n64(0.),
                }
            },
            loads,
            1,
        )
        .call_certain_within_bounds(t, x, &bounds)
    }

    fn movement(&self, prev_x: Config<T>, x: Config<T>, inverted: bool) -> N64 {
        movement(self.d, &self.switching_cost, prev_x, x, inverted)
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
    /// Vector of positive real constants resembling the switching cost of each dimension (strictly ascending).
    /// Dimensions must be _efficient_, i.e. there must not be dimensions with a higher switching and higher hitting cost than onether dimension.
    pub switching_cost: Vec<f64>,
    /// Time-independent cost of each dimension (strictly descending).
    pub hitting_cost: Vec<f64>,
    /// Non-negative load at each time step.
    pub load: Vec<T>,
}
impl_base_problem!(SmoothedLoadOptimization<T>);
impl<'a, T> Problem<T, (), DataCenterModelOutputFailure>
    for SmoothedLoadOptimization<T>
where
    T: Value<'a>,
{
    fn hit_cost(
        &self,
        t: i32,
        x: Config<T>,
    ) -> Cost<(), DataCenterModelOutputFailure> {
        if x.total() < self.load[t as usize - 1] {
            Cost::new(
                n64(f64::INFINITY),
                ModelOutput::Failure(
                    DataCenterModelOutputFailure::DemandExceedingSupply,
                ),
            )
        } else {
            FailableCost::raw(
                (0..self.d as usize)
                    .into_iter()
                    .map(|k| -> N64 {
                        let j: N64 = NumCast::from(x[k]).unwrap();
                        n64(self.hitting_cost[k]) * j
                    })
                    .sum(),
            )
        }
    }

    fn movement(&self, prev_x: Config<T>, x: Config<T>, inverted: bool) -> N64 {
        movement(self.d, &self.switching_cost, prev_x, x, inverted)
    }
}
pub type IntegralSmoothedLoadOptimization = SmoothedLoadOptimization<i32>;

fn movement<'a, T>(
    d: i32,
    switching_cost: &Vec<f64>,
    prev_x: Config<T>,
    x: Config<T>,
    inverted: bool,
) -> N64
where
    T: Value<'a>,
{
    (0..d as usize)
        .into_iter()
        .map(|k| -> N64 {
            let delta: N64 =
                NumCast::from(scalar_movement(x[k], prev_x[k], inverted))
                    .unwrap();
            n64(switching_cost[k]) * delta
        })
        .sum()
}
