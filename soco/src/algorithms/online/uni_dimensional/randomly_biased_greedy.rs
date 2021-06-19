use crate::config::{Config, FractionalConfig};
use crate::convex_optimization::find_minimizer;
use crate::online::{FractionalStep, Online, Step};
use crate::problem::FractionalSmoothedConvexOptimization;
use crate::result::{Error, Result};
use crate::schedule::FractionalSchedule;
use crate::utils::{assert, sample_uniform};

static DEFAULT_THETA: f64 = 1.;

/// Random number `r` representing bias.
pub type Memory = f64;

pub struct Options {
    /// Scaling factor for norm. `theta >= 1. Defaults to `1`.
    pub theta: Option<f64>,
}

/// Randomly Biased Greedy
pub fn rbg(
    o: &Online<FractionalSmoothedConvexOptimization<'_>>,
    xs: &mut FractionalSchedule,
    ms: &mut Vec<Memory>,
    options: &Options,
) -> Result<FractionalStep<()>> {
    assert(o.w == 0, Error::UnsupportedPredictionWindow)?;
    assert(o.p.d == 1, Error::UnsupportedProblemDimension)?;

    let theta = options.theta.unwrap_or(DEFAULT_THETA);

    let t = xs.t_end() + 1;
    let r = if t == 1 {
        sample_uniform(-1., 1.)
    } else {
        assert(ms.len() == 1, Error::MemoryShouldBePresent)?;
        ms[0]
    };

    let x = next(o, t, r, theta)?;
    Ok(Step(Config::single(x), None))
}

fn next(
    o: &Online<FractionalSmoothedConvexOptimization<'_>>,
    t: i32,
    r: f64,
    theta: f64,
) -> Result<f64> {
    let f = |raw_x: &[f64]| -> f64 {
        let x = Config::new(raw_x.to_vec());
        w(o, t - 1, theta, x.clone()).unwrap()
            + r * theta * (o.p.switching_cost)(x)
    };

    let (x, _) = find_minimizer(f, &o.p.bounds)?;
    Ok(x[0])
}

fn w(
    o: &Online<FractionalSmoothedConvexOptimization<'_>>,
    t: i32,
    theta: f64,
    x: FractionalConfig,
) -> Result<f64> {
    if t == 0 {
        Ok(theta * (o.p.switching_cost)(x))
    } else {
        let f = |raw_y: &[f64]| -> f64 {
            let y = Config::new(raw_y.to_vec());
            w(o, t - 1, theta, y.clone()).unwrap()
                + (o.p.hitting_cost)(t, y.clone()).unwrap()
                + theta * (o.p.switching_cost)(x.clone() - y)
        };

        let (y, _) = find_minimizer(f, &o.p.bounds)?;
        Ok(y[0])
    }
}
