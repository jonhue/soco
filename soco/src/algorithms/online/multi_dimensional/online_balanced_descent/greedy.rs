use crate::norm::NormFn;
use crate::algorithms::online::multi_dimensional::online_balanced_descent::meta::{obd, Options as MetaOptions};
use crate::algorithms::optimization::find_minimizer_of_hitting_cost;
use crate::config::FractionalConfig;
use crate::online::{FractionalStep, Online, Step};
use crate::problem::FractionalSmoothedConvexOptimization;
use crate::result::{Error, Result};
use crate::schedule::FractionalSchedule;
use crate::utils::assert;

static DEFAULT_MU: f64 = 1.;
static DEFAULT_GAMMA: f64 = 1.;

pub struct Options<'a> {
    /// Convexity parameter. Chosen such that `f_t(x) \geq f_t(v_t) + \frac{m}{2} \norm{x - v_t}_2^2` where `v_t` is the minimizer of `f_t`.
    pub m: f64,
    /// Controls the size of the step towards the minimizer. `mu > 0`. Defaults to `1`.
    pub mu: Option<f64>,
    /// Balance parameter in OBD. `gamma > 0`. Defaults to `1`.
    pub gamma: Option<f64>,
    /// Mirror map chosen based on the used norm.
    pub mirror_map: NormFn<'a, FractionalConfig>,
}

/// Greedy Online Balanced Descent
pub fn gobd(
    o: &Online<FractionalSmoothedConvexOptimization>,
    xs: &mut FractionalSchedule,
    _: &mut Vec<()>,
    options: &Options,
) -> Result<FractionalStep<()>> {
    assert(o.w == 0, Error::UnsupportedPredictionWindow)?;

    let mu = options.mu.unwrap_or(DEFAULT_MU);
    let gamma = options.gamma.unwrap_or(DEFAULT_GAMMA);

    let t = xs.t_end() + 1;

    let v = find_minimizer_of_hitting_cost(t, &o.p.hitting_cost, &o.p.bounds)?;
    let Step(y, _) = obd(
        o,
        xs,
        &mut vec![],
        &MetaOptions {
            l: gamma,
            mirror_map: options.mirror_map.clone(),
        },
    )?;

    let x = if mu * options.m.sqrt() >= 1. {
        v
    } else {
        mu * options.m.sqrt() * v + (1. - mu * options.m.sqrt()) * y
    };
    Ok(Step(x, None))
}
