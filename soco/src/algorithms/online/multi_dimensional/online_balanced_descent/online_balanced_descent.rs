use finitediff::FiniteDiff;
use nlopt::{Algorithm, Nlopt, Target};

use crate::algorithms::online::multi_dimensional::online_balanced_descent::mirror_map::MirrorMap;
use crate::config::Config;
use crate::cost::CostFn;
use crate::norm::NormFn;
use crate::online::{FractionalStep, Online, Step};
use crate::problem::FractionalSmoothedConvexOptimization;
use crate::result::{Error, Result};
use crate::schedule::FractionalSchedule;
use crate::utils::assert;
use crate::PRECISION;

pub struct Options<'a> {
    /// Determines the l-level set used in each step by the algorithm.
    pub l: f64,
    /// Mirror map chosen based on the used norm.
    pub mirror_map: MirrorMap<'a, Config<f64>>,
}

/// Online Balanced Descent (meta algorithm)
pub fn obd<'a>(
    o: &'a Online<FractionalSmoothedConvexOptimization>,
    xs: &mut FractionalSchedule,
    _: &mut Vec<()>,
    options: &Options<'a>,
) -> Result<FractionalStep<()>> {
    assert(o.w == 0, Error::UnsupportedPredictionWindow)?;

    let t = xs.t_end() + 1;
    let default_x = Config::repeat(0., o.p.d);
    let prev_x = if xs.is_empty() { &default_x } else { xs.now() };

    let x = bregman_projection(
        &options.mirror_map,
        &o.p.switching_cost,
        &o.p.hitting_cost,
        t,
        options.l,
        prev_x,
    )?;
    Ok(Step(x, None))
}

/// Bregman projection of `x` onto a convex `l`-sublevel set `K` of `f`.
///
/// `mirror_map` must be `m`-strongly convex and `M`-Lipschitz smooth for the norm function with fixed `m` and `M`.
fn bregman_projection<'a>(
    mirror_map: &MirrorMap<'a, Config<f64>>,
    norm: &NormFn<'a, Config<f64>>,
    f: &CostFn<'_, Config<f64>>,
    t: i32,
    l: f64,
    x: &Config<f64>,
) -> Result<Config<f64>> {
    let d = x.d() as usize;
    let objective_function = |y: &[f64],
                              _: Option<&mut [f64]>,
                              _: &mut ()|
     -> f64 {
        bregman_divergence(mirror_map, norm, Config::new(y.to_vec()), x.clone())
    };
    let mut y = vec![0.; d];
    let mut opt = Nlopt::new(
        Algorithm::Bobyqa,
        d,
        objective_function,
        Target::Minimize,
        (),
    );
    opt.set_xtol_rel(PRECISION)?;

    // `l`-sublevel set of `f`
    opt.add_inequality_constraint(
        |y: &[f64], _: Option<&mut [f64]>, _: &mut ()| -> f64 {
            f(t, Config::new(y.to_vec())).unwrap() - l
        },
        (),
        PRECISION,
    )?;

    opt.optimize(&mut y)?;
    Ok(Config::new(y))
}

/// Bregman divergence between `x` and `y`.
fn bregman_divergence<'a>(
    mirror_map: &MirrorMap<'a, Config<f64>>,
    norm: &NormFn<'a, Config<f64>>,
    x: Config<f64>,
    y: Config<f64>,
) -> f64 {
    let m = |x: &Vec<f64>| mirror_map(norm, Config::new(x.clone()));
    let mx = mirror_map(norm, x.clone());
    let my = mirror_map(norm, y.clone());
    let grad = Config::new(y.to_vec().central_diff(&m));
    mx - my - grad * (x - y)
}
