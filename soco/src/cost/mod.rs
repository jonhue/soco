//! Convex cost functions.

use std::sync::Arc;

pub mod data_center;

/// Cost function over time `t`. Must be total on `1<=t<=T`, `0<=j<=m`. May return `None` otherwise.
pub type CostFn<'a, T> = Arc<dyn Fn(i32, T) -> Option<f64> + 'a>;

/// Cost function (at time `t`). Must be total on `0<=j<=m`. May return `None` otherwise.
pub type SingleCostFn<'a, T> = Arc<dyn Fn(T) -> Option<f64> + 'a>;

/// Unifies a sequence of cost functions for different times `t` to a single cost function.
pub fn chain<'a, T>(fs: &'a Vec<SingleCostFn<'a, T>>) -> CostFn<'a, T> {
    Arc::new(move |t, j| {
        let i = t as usize - 1;
        if i <= fs.len() {
            fs[i](j)
        } else {
            None
        }
    })
}
