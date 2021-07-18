//! Convex cost functions.

use crate::config::Config;
use crate::utils::mean;
use crate::value::Value;
use noisy_float::prelude::*;
use num::NumCast;
use std::sync::Arc;

/// Cost function (from time `t`).
#[derive(Clone)]
pub struct SingleCostFn<'a, T>(
    Arc<dyn Fn(i32, T) -> Vec<R64> + Send + Sync + 'a>,
);
impl<'a, T> SingleCostFn<'a, T> {
    /// Creates a single cost function without uncertainty.
    pub fn certain(f: impl Fn(i32, T) -> R64 + Send + Sync + 'a) -> Self {
        Self::predictive(move |t, x| vec![f(t, x)])
    }

    /// Creates a single cost function with uncertainty.
    pub fn predictive(
        f: impl Fn(i32, T) -> Vec<R64> + Send + Sync + 'a,
    ) -> Self {
        Self(Arc::new(f))
    }

    /// Computes the hitting cost with an unbounded decision space.
    /// Returns mean if cost function returns a prediction.
    pub fn call_unbounded(&self, t: i32, x: T) -> R64 {
        assert!(
            t > 0,
            "Time slot of hitting cost must be positive (got {}).",
            t
        );
        let results = (self.0)(t, x);
        mean(results)
    }

    /// Computes the hitting cost.
    /// Returns mean if cost function returns a prediction.
    pub fn call<B>(&self, t: i32, x: T, bounds: &B) -> R64
    where
        B: DecisionSpace<'a, T>,
    {
        if bounds.within(&x) {
            self.call_unbounded(t, x)
        } else {
            r64(f64::INFINITY)
        }
    }

    /// Computes the hitting cost with an unbounded decision space.
    pub fn call_unbounded_predictive(&self, t: i32, x: T) -> Vec<R64> {
        assert!(
            t > 0,
            "Time slot of hitting cost must be positive (got {}).",
            t
        );
        let results = (self.0)(t, x);
        assert!(!results.is_empty());
        results
    }

    /// Computes the hitting cost.
    pub fn call_predictive<B>(&self, t: i32, x: T, bounds: &B) -> Vec<R64>
    where
        B: DecisionSpace<'a, T>,
    {
        if bounds.within(&x) {
            self.call_unbounded_predictive(t, x)
        } else {
            vec![r64(f64::INFINITY)]
        }
    }
}

/// Vector of cost functions that arrived over time. Individual cost functions may have different domains.
/// For example, in a predictive online setting, a cost function arriving at time `t` generally has the domain `[t, t + w]`.
#[derive(Clone)]
pub struct CostFn<'a, T>(Vec<SingleCostFn<'a, T>>);
impl<'a, T> CostFn<'a, T>
where
    T: Clone,
{
    /// Creates cost function from a vector of arriving cost functions `fs` beginning from some time `t_start >= 1`.
    pub fn new(t_start: i32, fs_: Vec<SingleCostFn<'a, T>>) -> Self {
        let mut fs: Vec<SingleCostFn<'a, T>> = vec![
            SingleCostFn::certain(
                |_, _| panic!("This time slot has no assigned cost function.")
            );
            t_start as usize - 1
        ];
        fs.extend(fs_);
        CostFn(fs)
    }

    /// Stretches a single cost function `f` over an interval from some time `t_start >= 1` to `t_end`.
    pub fn stretch(t_start: i32, t_end: i32, f: SingleCostFn<'a, T>) -> Self {
        Self::new(t_start, vec![f; (t_end - t_start + 1) as usize])
    }

    /// Creates a single cost function `f` at time `t`.
    pub fn single(t: i32, f: SingleCostFn<'a, T>) -> Self {
        Self::stretch(t, t, f)
    }

    /// Adds a new cost function which may return uncertain predictions.
    /// Must always return at least one sample (which corresponds to certainty).
    pub fn add(&mut self, f: SingleCostFn<'a, T>) {
        self.0.push(f)
    }

    /// Computes the hitting cost with an unbounded decision space.
    /// Returns mean if cost function returns a prediction.
    pub fn call_unbounded(&self, t: i32, x: T) -> R64 {
        assert!(
            t > 0,
            "Time slot of hitting cost must be positive (got {}).",
            t
        );
        self.get(t).call_unbounded(t, x)
    }

    /// Computes the hitting cost.
    /// Returns mean if cost function returns a prediction.
    pub fn call<B>(&self, t: i32, x: T, bounds: &B) -> R64
    where
        B: DecisionSpace<'a, T>,
    {
        if bounds.within(&x) {
            self.call_unbounded(t, x)
        } else {
            r64(f64::INFINITY)
        }
    }

    /// Computes the hitting cost with an unbounded decision space.
    pub fn call_unbounded_predictive(&self, t: i32, x: T) -> Vec<R64> {
        assert!(
            t > 0,
            "Time slot of hitting cost must be positive (got {}).",
            t
        );
        self.get(t).call_unbounded_predictive(t, x)
    }

    /// Computes the hitting cost.
    pub fn call_predictive<B>(&self, t: i32, x: T, bounds: &B) -> Vec<R64>
    where
        B: DecisionSpace<'a, T>,
    {
        if bounds.within(&x) {
            self.call_unbounded_predictive(t, x)
        } else {
            vec![r64(f64::INFINITY)]
        }
    }

    /// Returns the time of the latest cost function.
    pub fn now(&self) -> i32 {
        self.0.len() as i32
    }

    /// Finds the most recent version of the cost function which assigns a value to time slot `t`.
    ///
    /// If `t` is in the future (or now), the current cost function is used.
    /// If `t` is in the past, the cost function from time `t` is used.
    fn get(&self, t: i32) -> &SingleCostFn<'a, T> {
        assert!(
            self.now() > 0,
            "Cannot call a cost function without implementations."
        );
        let version = if t >= self.now() { self.now() } else { t };
        &self.0[version as usize - 1]
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
        for k in 0..self.len() {
            if x[k] < self[k].0 || x[k] > self[k].1 {
                return false;
            }
        }
        true
    }
}
impl<'a, T> DecisionSpace<'_, Config<T>> for Vec<T>
where
    T: Value<'a>,
{
    fn within(&self, x: &Config<T>) -> bool {
        assert!(x.d() == self.len() as i32);
        for k in 0..self.len() {
            if x[k] < NumCast::from(0).unwrap() || x[k] > self[k] {
                return false;
            }
        }
        true
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
