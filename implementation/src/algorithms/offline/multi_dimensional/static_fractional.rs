use crate::algorithms::offline::{OfflineOptions, PureOfflineResult};
use crate::config::Config;
use crate::model::{ModelOutputFailure, ModelOutputSuccess};
use crate::numerics::convex_optimization::{find_minimizer, WrappedObjective};
use crate::problem::{FractionalSmoothedConvexOptimization, Problem};
use crate::result::{Failure, Result};
use crate::schedule::Schedule;
use crate::utils::assert;

#[derive(Clone)]
struct ObjectiveData<'a, C, D> {
    p: FractionalSmoothedConvexOptimization<'a, C, D>,
    alpha: f64,
}

/// Algorithm computing the static fractional optimum.
pub fn static_fractional<C, D>(
    p: FractionalSmoothedConvexOptimization<'_, C, D>,
    _: (),
    OfflineOptions { inverted, alpha, l }: OfflineOptions,
) -> Result<PureOfflineResult<f64>>
where
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    assert(!inverted, Failure::UnsupportedInvertedCost)?;
    assert(
        l.is_none() || l == Some(0.),
        Failure::UnsupportedLConstrainedMovement,
    )?;

    let t_end = p.t_end;

    let bounds = p.bounds.clone();
    let objective =
        WrappedObjective::new(ObjectiveData { p, alpha }, |raw_x, data| {
            let x = Config::new(raw_x.to_vec());
            let xs = Schedule::repeat(x, data.p.t_end);
            data.p
                .alpha_unfair_objective_function(&xs, data.alpha)
                .unwrap()
                .cost
        });

    let (raw_x, _) = find_minimizer(objective, bounds);
    let x = Config::new(raw_x.to_vec());
    let xs = Schedule::repeat(x, t_end);
    Ok(PureOfflineResult { xs })
}
