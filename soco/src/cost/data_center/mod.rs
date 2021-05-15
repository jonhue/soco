//! Utilities to build cost functions for right-sizing data centers.

use num::ToPrimitive;
use std::sync::Arc;

use crate::cost::LoadCostFn;
use crate::value::Value;

use load::LoadFn;

pub mod load;

/// Given functions `f` that map the load of a server type to some cost metric:
/// returns a closure that given some workload `l` (at time `t`) for all servers,
/// distributes the load evenly across all active servers (at time `t`).
///
/// This behavior models the optimal dispatching rule of workload to all active servers.
pub fn load_balance<'a, T>(f: &'a Vec<LoadFn>) -> LoadCostFn<'a, T>
where
    T: Value + 'a,
{
    Arc::new(move |_, k, l| {
        Arc::new(move |x| {
            Some(
                ToPrimitive::to_f64(&x).unwrap()
                    * f[k as usize](
                        ToPrimitive::to_f64(&l).unwrap()
                            / ToPrimitive::to_f64(&x).unwrap(),
                    ),
            )
        })
    })
}
