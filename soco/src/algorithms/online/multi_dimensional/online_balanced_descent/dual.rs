use crate::algorithms::online::multi_dimensional::online_balanced_descent::{
    meta::{obd, Options as MetaOptions},
    MAX_L_FACTOR,
};
use crate::config::{Config, FractionalConfig};
use crate::norm::dual;
use crate::norm::NormFn;
use crate::numerics::bisection::bisection;
use crate::numerics::convex_optimization::find_minimizer_of_hitting_cost;
use crate::online::{FractionalStep, Online, Step};
use crate::problem::FractionalSmoothedConvexOptimization;
use crate::result::{Failure, Result};
use crate::schedule::FractionalSchedule;
use crate::utils::assert;
use finitediff::FiniteDiff;

pub struct Options<'a> {
    /// Balance parameter. `eta > 0`.
    pub eta: f64,
    /// Mirror map chosen based on the used norm.
    pub mirror_map: NormFn<'a, FractionalConfig>,
}

/// Dual Online Balanced Descent
pub fn dobd(
    o: &Online<FractionalSmoothedConvexOptimization>,
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
        find_minimizer_of_hitting_cost(t, &o.p.hitting_cost, &o.p.bounds)?.0,
    );
    let minimal_hitting_cost = o.p.hit_cost(t, v);

    let a = minimal_hitting_cost;
    let b = MAX_L_FACTOR * minimal_hitting_cost;
    let l = bisection((a, b), |l: f64| {
        balance_function(o, xs, &prev_x, t, l, options.eta, &options.mirror_map)
    })?;

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
    t: i32,
    l: f64,
    eta: f64,
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
    let f = |x: &Vec<f64>| o.p.hit_cost(t, Config::new(x.clone()));
    let m = |x: &Vec<f64>| mirror_map(Config::new(x.clone()));
    let distance = dual(
        &o.p.switching_cost,
        Config::new(x.to_vec().central_diff(&m))
            - Config::new(prev_x.to_vec().central_diff(&m)),
    )
    .unwrap();
    let hitting_cost = dual(
        &o.p.switching_cost,
        Config::new(x.to_vec().central_diff(&f)),
    )
    .unwrap();
    distance / hitting_cost - eta
}
