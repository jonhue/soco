//! Convex cost functions.

use crate::config::Config;
use crate::utils::mean;
use crate::value::Value;
use noisy_float::prelude::*;
use num::NumCast;
use std::collections::BTreeMap;
use std::sync::Arc;

/// Cost function (from time `t_start`).
#[derive(Clone)]
pub struct SingleCostFn<'a, T>(
    Arc<dyn Fn(i32, T) -> Vec<N64> + Send + Sync + 'a>,
);
impl<'a, T> SingleCostFn<'a, T> {
    /// Creates a single cost function without uncertainty.
    pub fn certain(f: impl Fn(i32, T) -> N64 + Send + Sync + 'a) -> Self {
        Self::predictive(move |t, x| vec![f(t, x)])
    }

    /// Creates a single cost function with uncertainty.
    pub fn predictive(
        f: impl Fn(i32, T) -> Vec<N64> + Send + Sync + 'a,
    ) -> Self {
        Self(Arc::new(f))
    }

    /// Returns mean if cost function returns a prediction.
    fn call_mean(&self, t_start: i32, t: i32, x: T) -> N64 {
        mean(self.call_predictive(t_start, t, x))
    }

    /// Computes certain cost.
    fn call_certain(&self, t_start: i32, t: i32, x: T) -> N64 {
        let results = self.call_predictive(t_start, t, x);
        assert!(results.len() == 1);
        results[0]
    }

    /// Computes uncertain cost.
    fn call_predictive(&self, t_start: i32, t: i32, x: T) -> Vec<N64> {
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

/// Cost functions that arrived over time. Individual cost functions may have different domains.
/// For example, in a predictive online setting, a cost function arriving at time `t` generally has the domain `[t, t + w]`.
#[derive(Clone)]
pub struct CostFn<'a, T>(BTreeMap<i32, SingleCostFn<'a, T>>);
impl<'a, T> CostFn<'a, T>
where
    T: Clone,
{
    /// Creates empty cost function.
    pub fn empty() -> Self {
        CostFn(BTreeMap::new())
    }

    /// Creates initial cost function from some time `t >= 1`.
    pub fn new(t: i32, f: SingleCostFn<'a, T>) -> Self {
        let mut fs = BTreeMap::new();
        fs.insert(t, f);
        CostFn(fs)
    }

    /// Adds a new cost function which may return uncertain predictions.
    /// Must always return at least one sample (which corresponds to certainty).
    pub fn add(&mut self, t: i32, f: SingleCostFn<'a, T>) {
        self.0.insert(t, f);
    }

    /// Returns mean if cost function returns a prediction.
    pub fn call_mean(&self, t: i32, x: T) -> N64 {
        let (&t_start, f) = self.get(t);
        f.call_mean(t_start, t, x)
    }

    /// Computes certain cost.
    pub fn call_certain(&self, t: i32, x: T) -> N64 {
        let (&t_start, f) = self.get(t);
        f.call_certain(t_start, t, x)
    }

    /// Computes uncertain cost.
    pub fn call_predictive(&self, t: i32, x: T) -> Vec<N64> {
        let (&t_start, f) = self.get(t);
        f.call_predictive(t_start, t, x)
    }

    /// Returns mean if cost function returns a prediction while ensuring that the given parameter is within the decision space.
    pub fn call_mean_within_bounds<B>(&self, t: i32, x: T, bounds: &B) -> N64
    where
        B: DecisionSpace<'a, T>,
    {
        if bounds.within(&x) {
            self.call_mean(t, x)
        } else {
            n64(f64::INFINITY)
        }
    }

    /// Computes certain cost while ensuring that the given parameter is within the decision space.
    pub fn call_certain_within_bounds<B>(&self, t: i32, x: T, bounds: &B) -> N64
    where
        B: DecisionSpace<'a, T>,
    {
        if bounds.within(&x) {
            self.call_certain(t, x)
        } else {
            n64(f64::INFINITY)
        }
    }

    /// Computes uncertain cost while ensuring that the given parameter is within the decision space.
    pub fn call_predictive_within_bounds<B>(
        &self,
        t: i32,
        x: T,
        bounds: &B,
    ) -> Vec<N64>
    where
        B: DecisionSpace<'a, T>,
    {
        if bounds.within(&x) {
            self.call_predictive(t, x)
        } else {
            vec![n64(f64::INFINITY)]
        }
    }

    /// Finds the most recent version of the cost function which assigns a value to time slot `t`.
    ///
    /// If `t` is in the future (or now), the current cost function is used.
    /// If `t` is in the past, the cost function from time `t` is used.
    ///
    /// Returns cost function and time slot of cost function.
    fn get(&self, t: i32) -> (&i32, &SingleCostFn<'a, T>) {
        self.0.range(1..=t).last().expect("Cost function does not have an implementation for the given time slot")
    }
}

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
