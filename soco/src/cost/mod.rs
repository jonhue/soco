//! Convex cost functions.

use nlopt::Algorithm;
use nlopt::Nlopt;
use nlopt::Target;
use num::{NumCast, ToPrimitive};
use std::sync::Arc;

use crate::utils::access;
use crate::value::Value;
use crate::PRECISION;

pub mod data_center;

/// Cost function over time `t`. Must be total on `1<=t<=T`, `0<=j<=m`. May return `None` otherwise.
pub type CostFn<'a, T> = Arc<dyn Fn(i32, T) -> Option<f64> + 'a>;

/// Cost function (at time `t`). Must be total on `0<=j<=m`. May return `None` otherwise.
pub type SingleCostFn<'a, T> = Arc<dyn Fn(T) -> Option<f64> + 'a>;

/// Collection of cost functions for some time, dimension, and load.
pub type LoadCostFn<'a, T> =
    Arc<dyn Fn(i32, i32, T) -> SingleCostFn<'a, T> + 'a>;

/// Converts a lazy cost function to a cost function that is total on `t`.
///
/// * `f` - Lazy cost function.
/// * `l` - Vector of loads for `0<=t<=T-1`. This vector can be updated online as `T` grows.
pub fn lazy<'a, T>(
    d: i32,
    f: LoadCostFn<'a, T>,
    ls: &'a Vec<T>,
) -> CostFn<'a, Vec<T>>
where
    T: Value,
{
    Arc::new(move |t, x| {
        let l = access(ls, t - 1)?;

        let objective_function =
            |zs: &[f64], _: Option<&mut [f64]>, _: &mut ()| -> f64 {
                x.iter()
                    .enumerate()
                    .map(|(k, &j)| {
                        let z = zs[k];
                        let prim_j = ToPrimitive::to_f64(&j)?;
                        let prim_l = ToPrimitive::to_f64(l)?;
                        if prim_j > 0. {
                            let l_frac = NumCast::from(prim_l * z)?;
                            f(t, k as i32, l_frac)(j)
                        } else if prim_j == 0. && prim_l * z > 0. {
                            Some(f64::INFINITY)
                        } else if prim_j == 0. && prim_l * z == 0. {
                            Some(0.)
                        } else {
                            None
                        }
                    })
                    .sum::<Option<f64>>()
                    .unwrap()
            };
        let mut zs = vec![1. / d as f64; d as usize];

        // minimize cost across all possible server to load matchings
        let mut opt = Nlopt::new(
            Algorithm::Bobyqa,
            d as usize,
            objective_function,
            Target::Minimize,
            (),
        );
        opt.set_lower_bound(0.).ok()?;
        opt.set_upper_bound(1.).ok()?;
        opt.set_xtol_rel(PRECISION).ok()?;

        opt.add_equality_constraint(
            |zs: &[f64], _: Option<&mut [f64]>, _: &mut ()| -> f64 {
                zs.iter().sum::<f64>() - 1.
            },
            (),
            PRECISION,
        )
        .ok()?;

        Some(opt.optimize(&mut zs).ok()?.1)
    })
}

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
