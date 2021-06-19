//! Utilities to build cost functions for right-sizing data centers.

use crate::config::Config;
use crate::convex_optimization::{minimize, Constraint};
use crate::cost::{CostFn, SingleCostFn};
use crate::utils::access;
use crate::value::Value;
use num::ToPrimitive;
use std::sync::Arc;

use load::Load;

pub mod load;
pub mod loss;

/// Collection of cost functions for some time, dimension, and load.
pub type LoadCostFn<'a, T> =
    Arc<dyn Fn(i32, usize, Load) -> SingleCostFn<'a, T> + 'a>;

/// Converts a load cost function to a cost function that is total on `t`.
///
/// * `d` - Number of dimensions.
/// * `e` - Number of load types.
/// * `f` - Load cost function.
/// * `l` - Vector of loads for `0<=t<=T-1`.
pub fn apply_loads<'a, T>(
    d: i32,
    e: i32,
    f: LoadCostFn<'a, T>,
    ls: Vec<Load>,
) -> CostFn<'a, Config<T>>
where
    T: Value + 'a,
{
    Arc::new(move |t, x| {
        let l = access(&ls, t - 1)?;

        let solver_d = (d * e) as usize;
        let bounds = vec![(0., 1.); solver_d];
        let objective = |zs: &[f64]| -> f64 {
            x.to_vec()
                .iter()
                .enumerate()
                .map(|(k, &j)| {
                    let z = zs[k * e as usize..k * e as usize + e as usize]
                        .to_vec();
                    let prim_j = ToPrimitive::to_f64(&j)?;
                    let sum_l = l.total();
                    let sum_z: f64 = z.iter().sum();

                    if prim_j > 0. {
                        let l_frac = l.clone() * z;
                        f(t, k, l_frac)(j)
                    } else if prim_j == 0. && sum_l * sum_z > 0. {
                        Some(f64::INFINITY)
                    } else if prim_j == 0. && sum_l * sum_z == 0. {
                        Some(0.)
                    } else {
                        None
                    }
                })
                .sum::<Option<f64>>()
                .unwrap()
        };

        // assigns each dimension a fraction of each load type
        let init = vec![1. / solver_d as f64; solver_d];

        // ensure that the fractions across all dimensions of each load type sum to `1`
        let mut equality_constraints: Vec<Constraint> = vec![];
        for j in 0..e as usize {
            equality_constraints.push(Arc::new(move |zs: &[f64]| -> f64 {
                let mut result = 0.;
                for k in 0..d as usize {
                    result += zs[k * j];
                }
                result - 1.
            }));
        }

        // minimize cost across all possible server to load matchings
        let (_, opt) = minimize(
            objective,
            &bounds,
            Some(init),
            vec![],
            equality_constraints,
        )
        .ok()?;
        Some(opt)
    })
}

/// Given functions `f` that map the load of a server type to some cost metric:
/// returns a closure that given some workload `l` (at time `t`) for all servers,
/// distributes the load evenly across all active servers (at time `t`).
///
/// This behavior models the optimal dispatching rule of workload to all active servers.
pub fn load_balance<'a, T>(
    f: Arc<dyn Fn(i32, usize, Load) -> Option<f64> + 'a>,
) -> LoadCostFn<'a, T>
where
    T: Value + 'a,
{
    Arc::new(move |t, k, l| {
        let f = f.clone();
        Arc::new(move |x| {
            Some(
                ToPrimitive::to_f64(&x).unwrap()
                    * f(t, k, l.clone() / ToPrimitive::to_f64(&x).unwrap())?,
            )
        })
    })
}
