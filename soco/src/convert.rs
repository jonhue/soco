//! Functions to convert between problem instances.

use crate::config::{Config, FractionalConfig, IntegralConfig};
use crate::cost::{Cost, CostFn, FailableCost, SingleCostFn};
use crate::distance::manhattan_scaled;
use crate::model::data_center::{
    DataCenterModelOutputFailure, DataCenterModelOutputSuccess,
};
use crate::model::{ModelOutput, ModelOutputFailure, ModelOutputSuccess};
use crate::problem::{
    FractionalSimplifiedSmoothedConvexOptimization,
    IntegralSimplifiedSmoothedConvexOptimization, Online, Problem,
    SimplifiedSmoothedConvexOptimization, SmoothedBalancedLoadOptimization,
    SmoothedConvexOptimization, SmoothedLoadOptimization,
};
use crate::schedule::{IntegralSchedule, Schedule};
use crate::utils::shift_time;
use crate::value::Value;
use crate::vec_wrapper::VecWrapper;
use noisy_float::prelude::*;
use num::{NumCast, ToPrimitive};

pub trait DiscretizableVector {
    /// Ceil all elements of a vector.
    fn ceil(&self) -> Vec<i32>;
    /// Floor all elements of a vector.
    fn floor(&self) -> Vec<i32>;
}

impl<'a, T> DiscretizableVector for Vec<T>
where
    T: Value<'a>,
{
    fn ceil(&self) -> Vec<i32> {
        self.iter().map(|&x| x.ceil()).collect()
    }

    fn floor(&self) -> Vec<i32> {
        self.iter().map(|&x| x.floor()).collect()
    }
}

pub trait CastableVector<T> {
    fn to(&self) -> Vec<T>;
}

impl<'a, T, U> CastableVector<T> for Vec<U>
where
    T: Value<'a>,
    U: Value<'a>,
{
    fn to(&self) -> Vec<T> {
        self.iter().map(|&x| NumCast::from(x).unwrap()).collect()
    }
}

pub trait DiscretizableCostFn<'a, C, D>
where
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    /// Discretize a certain fractional cost function.
    fn into_i(self) -> CostFn<'a, IntegralConfig, C, D>;
}

impl<'a, C, D> DiscretizableCostFn<'a, C, D>
    for CostFn<'a, FractionalConfig, C, D>
where
    C: ModelOutputSuccess + 'a,
    D: ModelOutputFailure + 'a,
{
    fn into_i(self) -> CostFn<'a, IntegralConfig, C, D> {
        CostFn::new(
            1,
            SingleCostFn::certain(move |t, x: IntegralConfig| {
                let f = self.clone();
                f.call_certain(t, x.to())
            }),
        )
    }
}

pub trait RelaxableCostFn<'a, C, D>
where
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    /// Relax a certain integral cost function to the fractional setting.
    fn into_f(self) -> CostFn<'a, FractionalConfig, C, D>;
}

impl<'a, C, D> RelaxableCostFn<'a, C, D> for CostFn<'a, IntegralConfig, C, D>
where
    C: ModelOutputSuccess + 'a,
    D: ModelOutputFailure + 'a,
{
    fn into_f(self) -> CostFn<'a, FractionalConfig, C, D> {
        CostFn::new(
            1,
            SingleCostFn::certain(move |t, x: FractionalConfig| {
                assert!(x.d() == 1, "cannot relax multidimensional problems");

                let j = n64(x[0]);
                if j.fract() == 0. {
                    self.call_certain(t, Config::single(j.to_i32().unwrap()))
                } else {
                    let lower = self.call_certain(
                        t,
                        Config::single(j.floor().to_i32().unwrap()),
                    );
                    let upper = self.call_certain(
                        t,
                        Config::single(j.ceil().to_i32().unwrap()),
                    );
                    Cost::new(
                        (j.ceil() - j) * lower.cost
                            + (j - j.floor()) * upper.cost,
                        ModelOutput::vertical_reduce(vec![
                            lower.output,
                            upper.output,
                        ]),
                    )
                }
            }),
        )
    }
}

pub trait DiscretizableProblem {
    type Output;

    /// Discretize a fractional problem instance.
    fn into_i(self) -> Self::Output;
}

impl<'a, C, D> DiscretizableProblem
    for FractionalSimplifiedSmoothedConvexOptimization<'a, C, D>
where
    C: ModelOutputSuccess + 'a,
    D: ModelOutputFailure + 'a,
{
    type Output = IntegralSimplifiedSmoothedConvexOptimization<'a, C, D>;

    fn into_i(self) -> IntegralSimplifiedSmoothedConvexOptimization<'a, C, D> {
        SimplifiedSmoothedConvexOptimization {
            d: self.d,
            t_end: self.t_end,
            bounds: self.bounds.floor(),
            switching_cost: self.switching_cost.clone(),
            hitting_cost: self.hitting_cost.into_i(),
        }
    }
}

pub trait RelaxableProblem {
    type Output;

    /// Relax an integral problem instance to the fractional setting.
    fn into_f(self) -> Self::Output;
}

impl<'a, C, D> RelaxableProblem
    for IntegralSimplifiedSmoothedConvexOptimization<'a, C, D>
where
    C: ModelOutputSuccess + 'a,
    D: ModelOutputFailure + 'a,
{
    type Output = FractionalSimplifiedSmoothedConvexOptimization<'a, C, D>;

    fn into_f(
        self,
    ) -> FractionalSimplifiedSmoothedConvexOptimization<'a, C, D> {
        SimplifiedSmoothedConvexOptimization {
            d: self.d,
            t_end: self.t_end,
            bounds: self.bounds.to(),
            switching_cost: self.switching_cost.clone(),
            hitting_cost: self.hitting_cost.into_f(),
        }
    }
}

impl<'a, T, C, D> SimplifiedSmoothedConvexOptimization<'a, T, C, D>
where
    T: Value<'a>,
    C: ModelOutputSuccess + 'a,
    D: ModelOutputFailure + 'a,
{
    /// Convert to an instance of Smoothed Convex Optimization.
    /// This assumes that time slots are added after this conversion.
    pub fn into_sco(self) -> SmoothedConvexOptimization<'a, T, C, D> {
        let bounds = self
            .bounds
            .iter()
            .map(|&u| (NumCast::from(0).unwrap(), u))
            .collect();
        let switching_cost = manhattan_scaled(self.switching_cost.clone());
        SmoothedConvexOptimization {
            d: self.d,
            t_end: self.t_end,
            bounds,
            switching_cost: switching_cost.clone(),
            hitting_cost: CostFn::new(
                1,
                SingleCostFn::certain(move |t: i32, x: Config<T>| {
                    if t == self.t_end {
                        let hitting_cost = self.hit_cost(t, x.clone());
                        Cost::new(
                            hitting_cost.cost + switching_cost(x),
                            hitting_cost.output,
                        )
                    } else {
                        self.hit_cost(t, x)
                    }
                }),
            ),
        }
    }
}

impl<'a, T> SmoothedBalancedLoadOptimization<'a, T>
where
    T: Value<'a>,
{
    /// Convert to an instance of Simplified Smoothed Convex Optimization.
    pub fn into_ssco(
        self,
    ) -> SimplifiedSmoothedConvexOptimization<
        'a,
        T,
        DataCenterModelOutputSuccess,
        DataCenterModelOutputFailure,
    > {
        SimplifiedSmoothedConvexOptimization {
            d: self.d,
            t_end: self.t_end,
            bounds: self.bounds.clone(),
            switching_cost: self.switching_cost.clone(),
            hitting_cost: CostFn::new(
                1,
                SingleCostFn::certain(move |t, x| self.clone().hit_cost(t, x)),
            ),
        }
    }
}

impl<'a, T> SmoothedLoadOptimization<T>
where
    T: Value<'a>,
{
    /// Convert instance to an instance of Smoothed Balanced-Load Optimization.
    pub fn into_sblo(self) -> SmoothedBalancedLoadOptimization<'a, T> {
        let hitting_cost = self
            .hitting_cost
            .iter()
            .map(|&c| {
                CostFn::new(
                    1,
                    SingleCostFn::certain(move |_, l| {
                        if l <= 1. {
                            FailableCost::raw(n64(c))
                        } else {
                            Cost::new(n64(f64::INFINITY), ModelOutput::Failure(DataCenterModelOutputFailure::SLOMaxUtilizationExceeded))
                        }
                    }),
                )
            })
            .collect();
        SmoothedBalancedLoadOptimization {
            d: self.d,
            t_end: self.t_end,
            bounds: self.bounds.clone(),
            switching_cost: self.switching_cost.clone(),
            hitting_cost,
            load: self.load.clone(),
        }
    }
}

impl<'a, P> Online<P>
where
    P: DiscretizableProblem,
{
    /// Discretize online problem.
    pub fn into_i(self) -> Online<P::Output> {
        Online {
            w: self.w,
            p: self.p.into_i(),
        }
    }
}

impl<'a, P> Online<P>
where
    P: RelaxableProblem,
{
    /// Relax online problem.
    pub fn into_f(self) -> Online<P::Output> {
        Online {
            w: self.w,
            p: self.p.into_f(),
        }
    }
}

impl<'a, T> Online<SmoothedLoadOptimization<T>>
where
    T: Value<'a>,
{
    pub fn into_sblo(self) -> Online<SmoothedBalancedLoadOptimization<'a, T>> {
        Online {
            w: self.w,
            p: self.p.into_sblo(),
        }
    }
}

impl<'a, T> Online<SmoothedBalancedLoadOptimization<'a, T>>
where
    T: Value<'a>,
{
    pub fn into_ssco(
        self,
    ) -> Online<
        SimplifiedSmoothedConvexOptimization<
            'a,
            T,
            DataCenterModelOutputSuccess,
            DataCenterModelOutputFailure,
        >,
    > {
        Online {
            w: self.w,
            p: self.p.into_ssco(),
        }
    }
}

impl<'a, T, C, D> Online<SimplifiedSmoothedConvexOptimization<'a, T, C, D>>
where
    T: Value<'a>,
    C: ModelOutputSuccess + 'a,
    D: ModelOutputFailure + 'a,
{
    pub fn into_sco(self) -> Online<SmoothedConvexOptimization<'a, T, C, D>> {
        Online {
            w: self.w,
            p: self.p.into_sco(),
        }
    }
}

pub trait DiscretizableConfig {
    /// Ceil all elements of a config.
    fn ceil(&self) -> IntegralConfig;
    /// Floor all elements of a config.
    fn floor(&self) -> IntegralConfig;
}

impl<'a, T> DiscretizableConfig for Config<T>
where
    T: Value<'a>,
{
    fn ceil(&self) -> IntegralConfig {
        Config::new(self.to_vec().ceil())
    }

    fn floor(&self) -> IntegralConfig {
        Config::new(self.to_vec().floor())
    }
}

pub trait CastableConfig<T> {
    fn to(&self) -> Config<T>;
}

impl<'a, T, U> CastableConfig<T> for Config<U>
where
    T: Value<'a>,
    U: Value<'a>,
{
    fn to(&self) -> Config<T> {
        Config::new(self.to_vec().to())
    }
}

pub trait DiscretizableSchedule {
    /// Discretize a schedule.
    fn to_i(&self) -> IntegralSchedule;
}

impl<'a, T> DiscretizableSchedule for Schedule<T>
where
    T: Value<'a>,
{
    fn to_i(&self) -> IntegralSchedule {
        self.iter().map(|x| x.ceil()).collect()
    }
}

pub trait CastableSchedule<T> {
    fn to(&self) -> Schedule<T>;
}

impl<'a, T, U> CastableSchedule<T> for Schedule<U>
where
    T: Value<'a>,
    U: Value<'a>,
{
    fn to(&self) -> Schedule<T> {
        self.iter().map(|x| x.to()).collect()
    }
}

pub trait ResettableCostFn<'a, T, C, D> {
    /// Shift a certain cost function to some new initial time `t_start` (a time _before_ first time slot).
    fn reset(&'a self, t_start: i32) -> CostFn<'a, T, C, D>;
}

impl<'a, T, C, D> ResettableCostFn<'a, T, C, D> for CostFn<'a, T, C, D>
where
    T: Clone,
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    fn reset(&'a self, t_start: i32) -> CostFn<'a, T, C, D> {
        CostFn::new(
            1,
            SingleCostFn::certain(move |t, j| {
                self.call_certain(shift_time(t, t_start + 1), j)
            }),
        )
    }
}

pub trait ResettableProblem<'a, T> {
    /// Shifts problem instance to some new initial time `t_start`.
    fn reset(&'a self, t_start: i32) -> Self;
}

impl<'a, T, C, D> ResettableProblem<'a, T>
    for SimplifiedSmoothedConvexOptimization<'a, T, C, D>
where
    T: Value<'a>,
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    fn reset(
        &'a self,
        t_start: i32,
    ) -> SimplifiedSmoothedConvexOptimization<'a, T, C, D> {
        SimplifiedSmoothedConvexOptimization {
            d: self.d,
            t_end: self.t_end - t_start,
            bounds: self.bounds.clone(),
            switching_cost: self.switching_cost.clone(),
            hitting_cost: self.hitting_cost.reset(t_start),
        }
    }
}
