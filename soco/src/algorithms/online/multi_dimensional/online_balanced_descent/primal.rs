use bacon_sci::roots::bisection;

use crate::algorithms::online::multi_dimensional::online_balanced_descent::{
    meta::{obd, Options as MetaOptions},
    MAX_ITERATIONS, MAX_L_FACTOR,
};
use crate::algorithms::optimization::find_minimizer;
use crate::config::{Config, FractionalConfig};
use crate::norm::NormFn;
use crate::online::{FractionalStep, Online, Step};
use crate::problem::FractionalSmoothedConvexOptimization;
use crate::result::{Error, Result};
use crate::schedule::FractionalSchedule;
use crate::utils::assert;
use crate::PRECISION;

pub struct Options<'a> {
    /// The movement cost is at most `beta` times the hitting cost. `beta > 0`.
    pub beta: f64,
    /// Mirror map chosen based on the used norm.
    pub mirror_map: NormFn<'a, FractionalConfig>,
}

/// Primal Online Balanced Descent
pub fn pobd(
    o: &Online<FractionalSmoothedConvexOptimization>,
    xs: &mut FractionalSchedule,
    _: &mut Vec<()>,
    options: &Options,
) -> Result<FractionalStep<()>> {
    assert(o.w == 0, Error::UnsupportedPredictionWindow)?;

    let t = xs.t_end() + 1;
    let prev_x = if xs.is_empty() {
        Config::repeat(0., o.p.d)
    } else {
        xs.now().clone()
    };

    let v = find_minimizer(t, &o.p.hitting_cost, &o.p.bounds)?;
    let dist = (o.p.switching_cost)(prev_x.clone() - v.clone());
    let minimal_hitting_cost =
        (o.p.hitting_cost)(t, v.clone()).ok_or(Error::CostFnMustBeTotal)?;
    if dist < options.beta * minimal_hitting_cost {
        return Ok(Step(v, None));
    }

    let a = minimal_hitting_cost;
    let b = MAX_L_FACTOR * minimal_hitting_cost;
    let l = bisection(
        (a, b),
        |l: f64| {
            balance_function(
                o,
                xs,
                &prev_x,
                l,
                options.beta,
                &options.mirror_map,
            )
        },
        PRECISION,
        MAX_ITERATIONS,
    )
    .map_err(Error::Bisection)?;

    obd(
        o,
        xs,
        &mut vec![],
        &MetaOptions {
            l,
            mirror_map: options.mirror_map.clone(),
        },
    )
}

fn balance_function(
    o: &Online<FractionalSmoothedConvexOptimization>,
    xs: &mut FractionalSchedule,
    prev_x: &FractionalConfig,
    l: f64,
    beta: f64,
    mirror_map: &NormFn<'_, FractionalConfig>,
) -> f64 {
    let Step(x, _) = obd(
        o,
        xs,
        &mut vec![],
        &MetaOptions {
            l,
            mirror_map: mirror_map.clone(),
        },
    )
    .unwrap();
    (o.p.switching_cost)(x - prev_x.clone()) - beta * l
}
