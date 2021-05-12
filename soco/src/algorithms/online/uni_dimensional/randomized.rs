use rand::{thread_rng, Rng};

use crate::algorithms::online::uni_dimensional::probabilistic::{
    probabilistic, Memory as ProbMemory,
};
use crate::convert::RelaxableSchedule;
use crate::online::{Online, OnlineSolution};
use crate::problem::ContinuousSmoothedConvexOptimization;
use crate::result::{Error, Result};
use crate::schedule::{DiscreteSchedule, Step};
use crate::utils::{assert, frac, project};

/// Continuous number of servers as determined by `bansal::det`; memory of `bansal::det`.
pub type Memory<'a> = (Step<f64>, ProbMemory<'a>);

/// Randomized Discrete Relaxation
///
/// Relax discrete problem to continuous problem before use!
pub fn randomized<'a>(
    o: &'a Online<ContinuousSmoothedConvexOptimization<'a>>,
    xs: &DiscreteSchedule,
    ms: &Vec<Memory<'a>>,
) -> Result<OnlineSolution<Step<i32>, Memory<'a>>> {
    assert(o.w == 0, Error::UnsupportedPredictionWindow)?;
    assert(o.p.d == 1, Error::UnsupportedProblemDimension)?;

    let prob_ms = ms.iter().map(|m| m.1.clone()).collect();
    let (y, prob_m) = probabilistic(o, &xs.to_f(), &prob_ms)?;

    let prev_x = if xs.is_empty() {
        0
    } else {
        xs[xs.len() - 1][0]
    };
    let prev_y = if ms.is_empty() {
        0.
    } else {
        ms[ms.len() - 1].0[0]
    };

    let x = next(prev_x, prev_y, y[0]);

    Ok((vec![x], (y, prob_m)))
}

fn next(prev_x: i32, prev_y: f64, y: f64) -> i32 {
    #[allow(clippy::collapsible_else_if)]
    // Number of active servers increases (or remains the same).
    if prev_y <= y {
        if prev_x == y.ceil() as i32 {
            prev_x
        } else {
            let prev_y_proj = project(prev_y, y.floor(), y.ceil());
            let p = (y - prev_y_proj) / (1. - frac(prev_y_proj));

            let r = sample_uniform();
            if r <= p {
                y.ceil() as i32
            } else {
                y.floor() as i32
            }
        }
    }
    // Number of active servers decreases.
    else {
        if prev_x == y.floor() as i32 {
            prev_x
        } else {
            let prev_y_proj = project(prev_y, y.floor(), y.ceil());
            let p = (prev_y_proj - y) / frac(prev_y_proj);

            let r = sample_uniform();
            if r <= p {
                y.floor() as i32
            } else {
                y.ceil() as i32
            }
        }
    }
}

fn sample_uniform() -> f64 {
    let mut rng = thread_rng();
    rng.gen_range(0.0..=1.0)
}
