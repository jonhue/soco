use crate::algorithms::offline::{OfflineOptions, PureOfflineResult};
use crate::config::Config;
use crate::model::{ModelOutputFailure, ModelOutputSuccess};
use crate::numerics::convex_optimization::{minimize, WrappedObjective};
use crate::problem::{FractionalSmoothedConvexOptimization, Problem};
use crate::result::{Failure, Result};
use crate::schedule::Schedule;
use crate::utils::assert;
use noisy_float::prelude::*;

struct ObjectiveData<'a, C, D> {
    p: FractionalSmoothedConvexOptimization<'a, C, D>,
    alpha: f64,
}

struct ConstraintData<'a, C, D> {
    p: FractionalSmoothedConvexOptimization<'a, C, D>,
    l: f64,
}

/// Convex Optimization
pub fn co<C, D>(
    p: FractionalSmoothedConvexOptimization<'_, C, D>,
    _: (),
    OfflineOptions { inverted, alpha, l }: OfflineOptions,
) -> Result<PureOfflineResult<f64>>
where
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
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
                .cost
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
