use crate::algorithms::offline::{OfflineOptions, PureOfflineResult};
use crate::config::IntegralConfig;
use crate::objective::Objective;
use crate::problem::IntegralSimplifiedSmoothedConvexOptimization;
use crate::result::{Failure, Result};
use crate::schedule::IntegralSchedule;
use crate::utils::assert;

/// Algorithm computing the static integral optimum.
///
/// Warning: do not use in practice, this algorithm is naive and has an exponential runtime.
pub fn static_integral(
    p: IntegralSimplifiedSmoothedConvexOptimization<'_>,
    _: (),
    OfflineOptions { inverted, alpha, l }: OfflineOptions,
) -> Result<PureOfflineResult<i32>> {
    assert(!inverted, Failure::UnsupportedInvertedCost)?;
    assert(l.is_none(), Failure::UnsupportedLConstrainedMovement)?;

    let (config, _) =
        check_configs(&p, alpha, 0, IntegralConfig::repeat(0, p.d))?;
    let xs = IntegralSchedule::new(vec![config; p.t_end as usize]);
    Ok(PureOfflineResult { xs })
}

fn check_configs(
    p: &IntegralSimplifiedSmoothedConvexOptimization<'_>,
    alpha: f64,
    k: usize,
    mut base_config: IntegralConfig,
) -> Result<(IntegralConfig, f64)> {
    if k < p.d as usize {
        let mut picked_config = base_config.clone();
        let mut picked_cost = f64::INFINITY;
        for j in 0..=p.bounds[k] {
            base_config[k] = j;
            let (config, cost) =
                check_configs(p, alpha, k + 1, base_config.clone())?;
            if cost < picked_cost {
                picked_config = config;
                picked_cost = cost;
            }
        }
        Ok((picked_config, picked_cost))
    } else {
        Ok((
            base_config.clone(),
            p.objective_function(&IntegralSchedule::new(vec![
                base_config;
                p.t_end as usize
            ]))?
            .raw(),
        ))
    }
}
