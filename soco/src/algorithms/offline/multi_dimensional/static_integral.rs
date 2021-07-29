use crate::algorithms::offline::{OfflineOptions, PureOfflineResult};
use crate::config::IntegralConfig;
use crate::model::{ModelOutputFailure, ModelOutputSuccess};
use crate::problem::{IntegralSmoothedConvexOptimization, Problem};
use crate::result::{Failure, Result};
use crate::schedule::IntegralSchedule;
use crate::utils::assert;

/// Algorithm computing the static integral optimum.
///
/// Warning: do not use in practice, this algorithm is naive and has an exponential runtime.
pub fn static_integral<C, D>(
    p: IntegralSmoothedConvexOptimization<'_, C, D>,
    _: (),
    OfflineOptions { inverted, alpha, l }: OfflineOptions,
) -> Result<PureOfflineResult<i32>>
where
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    assert(!inverted, Failure::UnsupportedInvertedCost)?;
    assert(
        l.is_none() || l == Some(0.),
        Failure::UnsupportedLConstrainedMovement,
    )?;

    let (config, _) =
        check_configs(&p, alpha, 0, IntegralConfig::repeat(0, p.d))?;
    let xs = IntegralSchedule::new(vec![config; p.t_end as usize]);
    Ok(PureOfflineResult { xs })
}

fn check_configs<C, D>(
    p: &IntegralSmoothedConvexOptimization<'_, C, D>,
    alpha: f64,
    k: usize,
    mut base_config: IntegralConfig,
) -> Result<(IntegralConfig, f64)>
where
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    if k < p.d as usize {
        let mut picked_config = base_config.clone();
        let mut picked_cost = f64::INFINITY;
        for j in p.bounds[k].0..=p.bounds[k].1 {
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
            .cost
            .raw(),
        ))
    }
}
