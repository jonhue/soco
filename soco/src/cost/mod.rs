//! Convex cost functions.

use crate::config::Config;
use crate::value::Value;
use num::NumCast;
use std::sync::Arc;

pub mod data_center;

/// Cost function over time `t`.
#[derive(Clone)]
pub struct CostFn<'a, T>(Arc<dyn Fn(i32, T) -> f64 + 'a>);
impl<'a, T> CostFn<'a, T> {
    pub fn new(f: impl Fn(i32, T) -> f64 + 'a) -> Self {
        CostFn(Arc::new(f))
    }

    /// Computes the hitting cost with an unbounded decision space.
    pub fn call_unbounded(&self, t: i32, x: T) -> f64 {
        assert!(t > 0, "Time slot of hitting cost must be positive.");
        (self.0)(t, x)
    }
}

/// Utility functions to evaluate hitting costs given a decision space.
pub trait CallableCostFn<'a, T, B> {
    fn call(&self, t: i32, x: T, bounds: B) -> f64;
}
impl<'a, T> CallableCostFn<'a, Config<T>, &Vec<(T, T)>>
    for CostFn<'a, Config<T>>
where
    T: Value,
{
    fn call(&self, t: i32, x: Config<T>, bounds: &Vec<(T, T)>) -> f64 {
        assert!(x.d() == bounds.len() as i32);
        for k in 0..bounds.len() {
            if x[k] < bounds[k].0 || x[k] > bounds[k].1 {
                return f64::INFINITY;
            }
        }
        self.call_unbounded(t, x)
    }
}
impl<'a, T> CallableCostFn<'a, Config<T>, &Vec<T>> for CostFn<'a, Config<T>>
where
    T: Value,
{
    fn call(&self, t: i32, x: Config<T>, bounds: &Vec<T>) -> f64 {
        assert!(x.d() == bounds.len() as i32);
        for k in 0..bounds.len() {
            if x[k] < NumCast::from(0).unwrap() || x[k] > bounds[k] {
                return f64::INFINITY;
            }
        }
        self.call_unbounded(t, x)
    }
}
impl<'a, T, U> CallableCostFn<'a, T, U> for CostFn<'a, T>
where
    T: Value,
    U: Value,
{
    fn call(&self, t: i32, x: T, bounds: U) -> f64 {
        if x < NumCast::from(0).unwrap() || x > NumCast::from(bounds).unwrap() {
            return f64::INFINITY;
        }
        self.call_unbounded(t, x)
    }
}

/// Cost function (at time `t`).
pub type SingleCostFn<'a, T> = Arc<dyn Fn(T) -> f64 + 'a>;

/// Unifies a sequence of cost functions for different times `t` to a single cost function.
pub fn chain<'a, T>(fs: &'a Vec<SingleCostFn<'a, T>>) -> CostFn<'a, T> {
    CostFn::new(move |t, j| {
        let i = t as usize - 1;
        assert!(
            i < fs.len(),
            "Chained hitting costs undefined for `t > {}` (got `{}`)",
            fs.len(),
            t
        );
        fs[i](j)
    })
}
