//! Convex cost functions.

use crate::config::Config;
use crate::model::{ModelOutput, ModelOutputFailure, ModelOutputSuccess};
use crate::utils::mean;
use crate::value::Value;
use noisy_float::prelude::*;
use num::NumCast;
use pyo3::{IntoPy, PyObject, Python};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::BTreeMap;
use std::iter::Sum;
use std::sync::Arc;

/// Result of cost computation.
#[derive(Clone, Debug)]
pub struct Cost<C, D> {
    pub cost: N64,
    pub output: ModelOutput<C, D>,
}
impl<C, D> Cost<C, D>
where
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    pub fn new(cost: N64, output: ModelOutput<C, D>) -> Self {
        Self { cost, output }
    }

    pub fn raw(cost: N64) -> Cost<(), D> {
        Cost {
            cost,
            output: ModelOutput::Success(()),
        }
    }

    pub fn mean(costs: Vec<Self>) -> Self {
        let (raw_costs, outputs) = costs
            .into_iter()
            .map(|Self { cost, output }| (cost, output))
            .unzip();
        Self {
            cost: mean(raw_costs),
            output: ModelOutput::reduce(outputs),
        }
    }

    pub fn to_raw(&self) -> (f64, ModelOutput<C, D>) {
        (self.cost.raw(), self.output.clone())
    }
}
impl<C, D> Default for Cost<C, D>
where
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    fn default() -> Self {
        Self {
            cost: n64(0.),
            output: ModelOutput::None,
        }
    }
}
impl<C, D> Sum for Cost<C, D>
where
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|result, value| Cost {
            cost: result.cost + value.cost,
            output: ModelOutput::reduce(vec![result.output, value.output]),
        })
        .unwrap_or_else(Cost::default)
    }
}
impl<C, D> Serialize for Cost<C, D>
where
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    #[inline]
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.to_raw().serialize(s)
    }
}
impl<'a, C, D> Deserialize<'a> for Cost<C, D>
where
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    #[inline]
    fn deserialize<T: Deserializer<'a>>(d: T) -> Result<Self, T::Error> {
        Deserialize::deserialize(d).map(|(cost, output)| Cost {
            cost: n64(cost),
            output,
        })
    }
}
impl<C, D> IntoPy<PyObject> for Cost<C, D>
where
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    fn into_py(self, py: Python) -> PyObject {
        self.to_raw().into_py(py)
    }
}

pub type FailableCost<D> = Cost<(), D>;
pub type RawCost = Cost<(), ()>;

/// Cost function (from time `t_start`).
#[derive(Clone)]
#[allow(clippy::type_complexity)]
pub struct SingleCostFn<'a, T, C, D>(
    Arc<dyn Fn(i32, T) -> Vec<Cost<C, D>> + Send + Sync + 'a>,
);
impl<'a, T, C, D> SingleCostFn<'a, T, C, D>
where
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    /// Creates a single cost function without uncertainty.
    pub fn certain(
        f: impl Fn(i32, T) -> Cost<C, D> + Send + Sync + 'a,
    ) -> Self {
        Self::predictive(move |t, x| vec![f(t, x)])
    }

    /// Creates a single cost function with uncertainty.
    pub fn predictive(
        f: impl Fn(i32, T) -> Vec<Cost<C, D>> + Send + Sync + 'a,
    ) -> Self {
        Self(Arc::new(f))
    }

    /// Returns mean if cost function returns a prediction.
    fn call_mean(&self, t_start: i32, t: i32, x: T) -> Cost<C, D> {
        Cost::mean(self.call_predictive(t_start, t, x))
    }

    /// Computes certain cost.
    fn call_certain(&self, t_start: i32, t: i32, x: T) -> Cost<C, D> {
        let results = self.call_predictive(t_start, t, x);
        assert!(results.len() == 1);
        results[0].clone()
    }

    /// Computes uncertain cost.
    fn call_predictive(&self, t_start: i32, t: i32, x: T) -> Vec<Cost<C, D>> {
        assert!(
            t >= t_start,
            "Time slot of hitting cost must be greater or equals to `t = {}` (got {}).",
            t_start,
            t
        );
        let results = (self.0)(t, x);
        if t == t_start {
            assert!(
                results.len() == 1,
                "Hitting costs must be certain for the current time slot."
            )
        }
        assert!(!results.is_empty());
        results
    }
}

pub type RawSingleCostFn<'a, T> = SingleCostFn<'a, T, (), ()>;

/// Cost functions that arrived over time. Individual cost functions may have different domains.
/// For example, in a predictive online setting, a cost function arriving at time `t` generally has the domain `[t, t + w]`.
#[derive(Clone)]
pub struct CostFn<'a, T, C, D>(BTreeMap<i32, SingleCostFn<'a, T, C, D>>);
impl<'a, T, C, D> CostFn<'a, T, C, D>
where
    T: Clone,
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    /// Creates empty cost function.
    pub fn empty() -> Self {
        CostFn(BTreeMap::new())
    }

    /// Creates initial cost function from some time `t >= 1`.
    pub fn new(t: i32, f: SingleCostFn<'a, T, C, D>) -> Self {
        let mut fs = BTreeMap::new();
        fs.insert(t, f);
        CostFn(fs)
    }

    /// Adds a new cost function which may return uncertain predictions.
    /// Must always return at least one sample (which corresponds to certainty).
    pub fn add(&mut self, t: i32, f: SingleCostFn<'a, T, C, D>) {
        self.0.insert(t, f);
    }

    /// Returns mean if cost function returns a prediction.
    pub fn call_mean(&self, t: i32, x: T) -> Cost<C, D> {
        let (&t_start, f) = self.get(t);
        f.call_mean(t_start, t, x)
    }

    /// Computes certain cost.
    pub fn call_certain(&self, t: i32, x: T) -> Cost<C, D> {
        let (&t_start, f) = self.get(t);
        f.call_certain(t_start, t, x)
    }

    /// Computes uncertain cost.
    pub fn call_predictive(&self, t: i32, x: T) -> Vec<Cost<C, D>> {
        let (&t_start, f) = self.get(t);
        f.call_predictive(t_start, t, x)
    }

    /// Returns mean if cost function returns a prediction while ensuring that the given parameter is within the decision space.
    pub fn call_mean_within_bounds<B>(
        &self,
        t: i32,
        x: T,
        bounds: &B,
    ) -> Cost<C, D>
    where
        B: DecisionSpace<'a, T>,
    {
        if bounds.within(&x) {
            self.call_mean(t, x)
        } else {
            Cost::new(
                n64(f64::INFINITY),
                ModelOutput::Failure(D::outside_decision_space()),
            )
        }
    }

    /// Computes certain cost while ensuring that the given parameter is within the decision space.
    pub fn call_certain_within_bounds<B>(
        &self,
        t: i32,
        x: T,
        bounds: &B,
    ) -> Cost<C, D>
    where
        B: DecisionSpace<'a, T>,
    {
        if bounds.within(&x) {
            self.call_certain(t, x)
        } else {
            Cost::new(
                n64(f64::INFINITY),
                ModelOutput::Failure(D::outside_decision_space()),
            )
        }
    }

    /// Computes uncertain cost while ensuring that the given parameter is within the decision space.
    pub fn call_predictive_within_bounds<B>(
        &self,
        t: i32,
        x: T,
        bounds: &B,
    ) -> Vec<Cost<C, D>>
    where
        B: DecisionSpace<'a, T>,
    {
        if bounds.within(&x) {
            self.call_predictive(t, x)
        } else {
            vec![Cost::new(
                n64(f64::INFINITY),
                ModelOutput::Failure(D::outside_decision_space()),
            )]
        }
    }

    /// Finds the most recent version of the cost function which assigns a value to time slot `t`.
    ///
    /// If `t` is in the future (or now), the current cost function is used.
    /// If `t` is in the past, the cost function from time `t` is used.
    ///
    /// Returns cost function and time slot of cost function.
    fn get(&self, t: i32) -> (&i32, &SingleCostFn<'a, T, C, D>) {
        self.0.range(1..=t).last().expect("Cost function does not have an implementation for the given time slot")
    }
}

pub type FailableCostFn<'a, T, D> = CostFn<'a, T, (), D>;
pub type RawCostFn<'a, T> = CostFn<'a, T, (), ()>;

pub trait DecisionSpace<'a, T> {
    /// Checks whether the given configuration is within the decision space.
    fn within(&self, x: &T) -> bool;
}
impl<'a, T> DecisionSpace<'_, Config<T>> for Vec<(T, T)>
where
    T: Value<'a>,
{
    fn within(&self, x: &Config<T>) -> bool {
        assert!(x.d() == self.len() as i32);
        self.iter()
            .enumerate()
            .all(|(k, &(l, u))| x[k] >= l && x[k] <= u)
    }
}
impl<'a, T> DecisionSpace<'_, Config<T>> for Vec<T>
where
    T: Value<'a>,
{
    fn within(&self, x: &Config<T>) -> bool {
        assert!(x.d() == self.len() as i32);
        self.iter()
            .enumerate()
            .all(|(k, &u)| x[k] >= NumCast::from(0).unwrap() && x[k] <= u)
    }
}
impl<'a, T, U> DecisionSpace<'a, T> for U
where
    T: Value<'a>,
    U: Value<'a>,
{
    fn within(&self, x: &T) -> bool {
        if *x < NumCast::from(0).unwrap() || *x > NumCast::from(*self).unwrap()
        {
            return false;
        }
        true
    }
}
