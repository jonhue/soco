use crate::algorithms::offline::{OfflineOptions, PureOfflineResult};
use crate::config::Config;
use crate::numerics::convex_optimization::{minimize, WrappedObjective};
use crate::objective::Objective;
use crate::problem::FractionalSmoothedConvexOptimization;
use crate::result::{Failure, Result};
use crate::schedule::Schedule;
use crate::utils::assert;
use noisy_float::prelude::*;

struct ObjectiveData<'a> {
    p: FractionalSmoothedConvexOptimization<'a>,
    alpha: f64,
}

struct ConstraintData<'a> {
    p: FractionalSmoothedConvexOptimization<'a>,
    l: f64,
}

/// Convex Optimization
pub fn co(
    p: FractionalSmoothedConvexOptimization<'_>,
    _: (),
    OfflineOptions { inverted, alpha, l }: OfflineOptions,
) -> Result<PureOfflineResult<f64>> {
    assert(!inverted, Failure::UnsupportedInvertedCost)?;

    let d = p.d;
    let t_end = p.t_end;

    let (lower, upper): (Vec<_>, Vec<_>) = p.bounds.iter().cloned().unzip();
    let extended_lower = Schedule::build_raw(p.t_end, &Config::new(lower));
    let extended_upper = Schedule::build_raw(p.t_end, &Config::new(upper));
    let bounds = extended_lower
        .into_iter()
        .zip(extended_upper.into_iter())
        .collect();

    let objective = WrappedObjective::new(
        ObjectiveData {
            p: p.clone(),
            alpha,
        },
        |raw_xs, data| {
            let xs = Schedule::from_raw(data.p.d, data.p.t_end, raw_xs);
            data.p
                .alpha_unfair_objective_function(&xs, data.alpha)
                .unwrap()
        },
    );

    // l-constrained movement
    let constraints = match l {
        Some(l) => {
            vec![WrappedObjective::new(
                ConstraintData { p, l },
                |raw_xs, data| {
                    let xs = Schedule::from_raw(data.p.d, data.p.t_end, raw_xs);
                    data.p.total_movement(&xs, false).unwrap() - n64(data.l)
                },
            )]
        }
        None => vec![],
    };

    let (raw_xs, _) = minimize(objective, bounds, None, constraints)?;
    let xs = Schedule::from_raw(d, t_end, &raw_xs);
    Ok(PureOfflineResult { xs })
}
