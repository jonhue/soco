use crate::algorithms::online::multi_dimensional::online_balanced_descent::{
    meta::{obd, Options as MetaOptions},
    MAX_L_FACTOR,
};
use crate::algorithms::online::{FractionalStep, Step};
use crate::config::{Config, FractionalConfig};
use crate::norm::NormFn;
use crate::numerics::convex_optimization::find_minimizer_of_hitting_cost;
use crate::numerics::roots::find_root;
use crate::problem::{FractionalSmoothedConvexOptimization, Online};
use crate::result::{Failure, Result};
use crate::schedule::FractionalSchedule;
use crate::utils::assert;

pub struct Options<'a> {
    /// The movement cost is at most `beta` times the hitting cost. `beta > 0`.
    pub beta: f64,
    /// Mirror map chosen based on the used norm.
    pub mirror_map: NormFn<'a, f64>,
}

/// Primal Online Balanced Descent
pub fn pobd(
    o: Online<FractionalSmoothedConvexOptimization>,
    xs: &mut FractionalSchedule,
    _: &mut Vec<()>,
    options: &Options,
) -> Result<FractionalStep<()>> {
    assert(o.w == 0, Failure::UnsupportedPredictionWindow(o.w))?;

    let t = xs.t_end() + 1;
    let prev_x = if xs.is_empty() {
        Config::repeat(0., o.p.d)
    } else {
        xs.now()
    };

    let v = Config::new(
        find_minimizer_of_hitting_cost(
            t,
            o.p.hitting_cost.clone(),
            o.p.bounds.clone(),
        )?
        .0,
    );
    let dist = (o.p.switching_cost)(prev_x.clone() - v.clone()).raw();
    let minimal_hitting_cost = o.p.hit_cost(t, v.clone()).raw();
    if dist < options.beta * minimal_hitting_cost {
        return Ok(Step(v, None));
    }

    let a = minimal_hitting_cost;
    let b = MAX_L_FACTOR * minimal_hitting_cost;
    let l = find_root((a, b), |l: f64| {
        balance_function(&o, xs, &prev_x, l, options.beta, &options.mirror_map)
    })?
    .raw();

    obd(
        o,
        xs,
        &mut vec![],
        MetaOptions {
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
    mirror_map: &NormFn<'_, f64>,
) -> f64 {
    let Step(x, _) = obd(
        o.clone(),
        xs,
        &mut vec![],
        MetaOptions {
            l,
            mirror_map: mirror_map.clone(),
        },
    )
    .unwrap();
    (o.p.switching_cost)(x - prev_x.clone()).raw() - beta * l
}
