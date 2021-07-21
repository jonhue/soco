use crate::algorithms::offline::{OfflineOptions, PureOfflineResult};
use crate::config::Config;
use crate::numerics::convex_optimization::{minimize, Constraint};
use crate::objective::Objective;
use crate::problem::FractionalSmoothedConvexOptimization;
use crate::result::{Failure, Result};
use crate::schedule::Schedule;
use crate::utils::assert;
use noisy_float::prelude::*;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    ParallelIterator,
};
use std::sync::Arc;

/// Convex Optimization
pub fn co(
    p: FractionalSmoothedConvexOptimization<'_>,
    _: (),
    OfflineOptions { inverted, alpha, l }: OfflineOptions,
) -> Result<PureOfflineResult<f64>> {
    assert(!inverted, Failure::UnsupportedInvertedCost)?;

    let d = p.d;
    let t_end = p.t_end;

    let (lower, upper): (Vec<_>, Vec<_>) = p.bounds.par_iter().cloned().unzip();
    let extended_lower = Schedule::build_raw(t_end, &Config::new(lower));
    let extended_upper = Schedule::build_raw(t_end, &Config::new(upper));
    let bounds = extended_lower
        .into_par_iter()
        .zip(extended_upper.into_par_iter())
        .collect();

    let objective = |raw_xs: &[f64]| {
        let xs = Schedule::from_raw(d, t_end, raw_xs);
        p.alpha_unfair_objective_function(&xs, alpha).unwrap()
    };

    // l-constrained movement
    let inequality_constraints: Vec<Constraint> = match l {
        Some(l) => {
            let p = p.clone();
            vec![Arc::new(move |raw_xs: &[f64]| -> N64 {
                let xs = Schedule::from_raw(d, t_end, raw_xs);
                p.movement(&xs, false).unwrap() - n64(l)
            })]
        }
        None => vec![],
    };

    let (raw_xs, _) =
        minimize(objective, &bounds, None, inequality_constraints, vec![])?;
    let xs = Schedule::from_raw(d, t_end, &raw_xs);
    Ok(PureOfflineResult { xs })
}
