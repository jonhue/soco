use bacon_sci::roots::bisection;

use crate::algorithms::optimization::find_minimizer;
use crate::algorithms::online::multi_dimensional::online_balanced_descent::{MAX_L_FACTOR, MAX_ITERATIONS};
use crate::algorithms::online::multi_dimensional::online_balanced_descent::mirror_map::MirrorMap;
use crate::algorithms::online::multi_dimensional::online_balanced_descent::online_balanced_descent::{obd, Options as MetaOptions};
use crate::config::Config;
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
    pub mirror_map: MirrorMap<'a, Config<f64>>,
}

/// Primal Online Balanced Descent
pub fn pobd<'a>(
    o: &'a Online<FractionalSmoothedConvexOptimization>,
    xs: &mut FractionalSchedule,
    _: &mut Vec<()>,
    options: &Options<'a>,
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
            let Step(x, _) = obd(
                o,
                xs,
                &mut vec![],
                &MetaOptions {
                    l,
                    mirror_map: options.mirror_map.clone(),
                },
            )
            .unwrap();
            (o.p.switching_cost)(x - prev_x.clone()) - options.beta * l
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