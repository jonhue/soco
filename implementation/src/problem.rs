//! Abstract definition of problem variants.

use crate::config::Config;
use crate::cost::{Cost, CostFn, FailableCost, FailableCostFn, RawCost};
use crate::distance::NormFn;
use crate::model::data_center::loads::{
    apply_loads_over_time, LoadFractions, LoadProfile,
};
use crate::model::data_center::safe_balancing;
use crate::model::data_center::{
    DataCenterModelOutputFailure, DataCenterModelOutputSuccess,
    DataCenterObjective, IntermediateObjective,
};
use crate::model::{ModelOutput, ModelOutputFailure, ModelOutputSuccess};
use crate::result::Result;
use crate::schedule::Schedule;
use crate::utils::pos;
use crate::value::Value;
use crate::vec_wrapper::VecWrapper;
use crate::verifiers::VerifiableProblem;
use noisy_float::prelude::*;
use num::NumCast;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

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

/// Trait providing objective function, hitting cost, and movement cost.
pub trait Problem<T, C, D>: BaseProblem + Sync
where
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    /// Hitting cost.
    fn hit_cost(&self, t: i32, x: Config<T>) -> Cost<C, D>;

    /// Movement cost.
    fn movement(&self, prev_x: Config<T>, x: Config<T>, inverted: bool) -> N64;

    /// Objective function. Calculates the cost of a schedule.
    fn objective_function<'a>(&self, xs: &Schedule<T>) -> Result<Cost<C, D>>
    where
        T: Value<'a>,
    {
        let default = self._default_config();
        self._objective_function_with_default(
            xs,
            &default,
            1.,
            false,
            self.t_end(),
        )
    }

    /// Inverted Objective function. Calculates the cost of a schedule. Pays the
    /// switching cost for powering down rather than powering up.
    fn inverted_objective_function<'a>(
        &self,
        xs: &Schedule<T>,
    ) -> Result<Cost<C, D>>
    where
        T: Value<'a>,
    {
        let default = self._default_config();
        self._objective_function_with_default(
            xs,
            &default,
            1.,
            true,
            self.t_end(),
        )
    }

    /// $\alpha$-unfair Objective function. Calculates the cost of a schedule.
    fn alpha_unfair_objective_function<'a>(
        &self,
        xs: &Schedule<T>,
        alpha: f64,
    ) -> Result<Cost<C, D>>
    where
        T: Value<'a>,
    {
        let default = self._default_config();
        self._objective_function_with_default(
            xs,
            &default,
            alpha,
            false,
            self.t_end(),
        )
    }

    /// Objective function starting from an initial configuration other than $\mathbf{0}$.
    fn objective_function_with_default<'a>(
        &self,
        xs: &Schedule<T>,
        default: &Config<T>,
    ) -> Result<Cost<C, D>>
    where
        T: Value<'a>,
    {
        self._objective_function_with_default(
            xs,
            default,
            1.,
            false,
            self.t_end(),
        )
    }

    fn _objective_function_with_default<'a>(
        &self,
        xs: &Schedule<T>,
        default: &Config<T>,
        alpha: f64,
        inverted: bool,
        t_end: i32,
    ) -> Result<Cost<C, D>>
    where
        T: Value<'a>,
    {
        Ok(sum_over_schedule(t_end, xs, default, |t, prev_x, x| {
            let hitting_cost = if t > self.t_end() {
                Default::default()
            } else {
                self.hit_cost(t, x.clone())
            };
            Cost::new(
                hitting_cost.cost
                    + n64(alpha) * self.movement(prev_x, x, inverted),
                hitting_cost.output,
            )
        }))
    }

    /// Movement in the decision space.
    fn total_movement<'a>(
        &self,
        xs: &Schedule<T>,
        inverted: bool,
    ) -> Result<N64>
    where
        T: Value<'a>,
    {
        let default = self._default_config();
        Ok(
            sum_over_schedule(self.t_end(), xs, &default, |_, prev_x, x| {
                RawCost::raw(self.movement(prev_x, x, inverted))
            })
            .cost,
        )
    }

    fn _default_config<'a>(&self) -> Config<T>
    where
        T: Value<'a>,
    {
        Config::repeat(NumCast::from(0).unwrap(), self.d())
    }
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
    P: Problem<T, C, D>,
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
    U: Default,
{
    fn default(_: &P) -> Self {
        U::default()
    }
}

/// Gives type a default value which may depend on an online problem instance.
pub trait DefaultGivenOnlineProblem<T, P, C, D>
where
    P: Problem<T, C, D>,
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    fn default(o: &Online<P>) -> Self;
}
impl<T, P, C, D, U> DefaultGivenOnlineProblem<T, P, C, D> for U
where
    P: Problem<T, C, D>,
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
    U: Default,
{
    fn default(_: &Online<P>) -> Self {
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
    /// i.e. at time $t$ $t_end$ should be set to $t + w$.
    pub w: i32,
}

/// Smoothed Convex Optimization (SCO).
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
    pub switching_cost: NormFn<T>,
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
            .call_mean_within_bounds(t, x, &self.bounds)
    }

    fn movement(&self, prev_x: Config<T>, x: Config<T>, inverted: bool) -> N64 {
        assert!(!inverted, "Unsupported inverted movement.");

        (self.switching_cost)(x - prev_x)
    }
}
pub type IntegralSmoothedConvexOptimization<'a, C, D> =
    SmoothedConvexOptimization<'a, i32, C, D>;
pub type FractionalSmoothedConvexOptimization<'a, C, D> =
    SmoothedConvexOptimization<'a, f64, C, D>;

/// Simplified Smoothed Convex Optimization (SSCO).
///
/// * decision space is lower bounded by $\mathbf{0}$
/// * movement costs are a dimension-dependently scaled Manhattan distance
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
            .call_mean_within_bounds(t, x, &self.bounds)
    }

    fn movement(&self, prev_x: Config<T>, x: Config<T>, inverted: bool) -> N64 {
        scaled_movement(&self.switching_cost, &x, &prev_x, inverted)
    }
}
pub type IntegralSimplifiedSmoothedConvexOptimization<'a, C, D> =
    SimplifiedSmoothedConvexOptimization<'a, i32, C, D>;
pub type FractionalSimplifiedSmoothedConvexOptimization<'a, C, D> =
    SimplifiedSmoothedConvexOptimization<'a, f64, C, D>;

/// Smoothed Balanced-Load Optimization (SBLO).
///
/// * SSCO
/// * hitting costs are computed by _balancing_ incoming loads across all dimensions each of which is described by a convex cost function
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
                (0..self.d as usize)
                    .map(|k| {
                        let total_load = zs.select_loads(lambda, k)[0];
                        let x = NumCast::from(x_[k]).unwrap();
                        Ok(DataCenterObjective::new(
                            safe_balancing(x, total_load, || {
                                Ok(x * self.hitting_cost[k]
                                    .call_mean(t, (total_load / x).raw())
                                    .cost)
                            })?,
                            n64(0.),
                        ))
                    })
                    .sum::<IntermediateObjective>()
            },
            loads,
            1,
        )
        .call_certain_within_bounds(t, x, &bounds)
    }

    fn movement(&self, prev_x: Config<T>, x: Config<T>, inverted: bool) -> N64 {
        scaled_movement(&self.switching_cost, &x, &prev_x, inverted)
    }
}
pub type IntegralSmoothedBalancedLoadOptimization<'a> =
    SmoothedBalancedLoadOptimization<'a, i32>;

/// Smoothed Load Optimization (SLO).
///
/// * SSCO
/// * hitting costs are time independent and linear in some incoming load
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
        scaled_movement(&self.switching_cost, &x, &prev_x, inverted)
    }
}
pub type IntegralSmoothedLoadOptimization = SmoothedLoadOptimization<i32>;

fn sum_over_schedule<'a, T, C, D>(
    t_end: i32,
    xs: &Schedule<T>,
    default: &Config<T>,
    f: impl Fn(i32, Config<T>, Config<T>) -> Cost<C, D> + Send + Sync,
) -> Cost<C, D>
where
    T: Value<'a>,
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    (1..=t_end)
        .into_par_iter()
        .map(|t| {
            let prev_x = xs.get(t - 1).unwrap_or(default).clone();
            let x = xs.get(t).unwrap_or(default).clone();
            f(t, prev_x, x)
        })
        .sum()
}

/// Movement between two values (one dimensional points).
pub fn scalar_movement<'a, T>(j: T, prev_j: T, inverted: bool) -> T
where
    T: Value<'a>,
{
    pos(if inverted { prev_j - j } else { j - prev_j })
}

/// Movement between two configurations (points in the decision space).
pub fn movement<'a, T>(
    x: &Config<T>,
    prev_x: &Config<T>,
    inverted: bool,
) -> Config<T>
where
    T: Value<'a>,
{
    (0..x.d() as usize)
        .into_iter()
        .map(|k| scalar_movement(x[k], prev_x[k], inverted))
        .collect()
}

/// Movement scaled by dimension-dependent switching costs.
pub fn scaled_movement<'a, T>(
    switching_cost: &Vec<f64>,
    x: &Config<T>,
    prev_x: &Config<T>,
    inverted: bool,
) -> N64
where
    T: Value<'a>,
{
    movement(x, prev_x, inverted)
        .iter()
        .map(|&delta| -> N64 { NumCast::from(delta).unwrap() })
        .enumerate()
        .map(|(k, delta)| -> N64 { n64(switching_cost[k]) * delta })
        .sum()
}
