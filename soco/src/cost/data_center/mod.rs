//! Utilities to build cost functions for right-sizing data centers.

use num::ToPrimitive;
use std::sync::Arc;

use crate::cost::LazyCostFn;

use load::LoadFn;

pub mod load;

/// Given a function `f` that maps the load of a server to some cost metric:
/// returns a closure that given some workload `l` (at time `t`) for all servers,
/// distributes the load evenly across all active servers (at time `t`).
///
/// This behavior models the optimal dispatching rule of workload to all active servers.
pub fn load_balance<'a, T>(f: &'a LoadFn) -> LazyCostFn<'a, T>
where
    T: Clone + ToPrimitive + 'a,
{
    Arc::new(move |l| {
        Arc::new(move |x| {
            Some(
                ToPrimitive::to_f64(&x).unwrap()
                    * f(ToPrimitive::to_f64(&l).unwrap()
                        / ToPrimitive::to_f64(&x).unwrap()),
            )
        })
    })
}
