use crate::algorithms::online::uni_dimensional::probabilistic::{
    probabilistic, Memory as RelaxationMemory, Options as RelaxationOptions,
};
use crate::breakpoints::Breakpoints;
use crate::config::{Config, FractionalConfig};
use crate::convert::RelaxableSchedule;
use crate::online::{IntegralStep, Online, OnlineAlgorithm, Step};
use crate::problem::FractionalSimplifiedSmoothedConvexOptimization;
use crate::result::{Failure, Result};
use crate::schedule::IntegralSchedule;
use crate::utils::{assert, frac, project, sample_uniform};

/// Memory.
#[derive(Clone)]
pub struct Memory<'a> {
    /// Fractional number of servers determined by fractional relaxation.
    y: FractionalConfig,
    /// Memory of relaxation.
    relaxation_m: Option<RelaxationMemory<'a>>,
}
impl Default for Memory<'_> {
    fn default() -> Self {
        Memory {
            y: Config::single(0.),
            relaxation_m: None,
        }
    }
}

/// Randomized Integral Relaxation
///
/// Relax discrete problem to fractional problem before use!
pub fn randomized<'a>(
    o: Online<FractionalSimplifiedSmoothedConvexOptimization<'a>>,
    _: i32,
    xs: &IntegralSchedule,
    prev_m: Memory<'a>,
    _: &(),
) -> Result<IntegralStep<Memory<'a>>> {
    assert(o.w == 0, Failure::UnsupportedPredictionWindow(o.w))?;
    assert(o.p.d == 1, Failure::UnsupportedProblemDimension(o.p.d))?;
    assert(o.p.bounds[0].fract() == 0., Failure::MustBeRelaxedProblem)?;

    let relaxation_options = RelaxationOptions {
        breakpoints: Breakpoints::grid(),
    };
    let Step(y, relaxation_m) = probabilistic.next(
        o,
        &xs.to_f(),
        prev_m.relaxation_m,
        &relaxation_options,
    )?;

    let prev_x = xs.now_with_default(Config::single(0))[0];
    let prev_y = prev_m.y[0];

    let x = next(prev_x, prev_y, y[0]);
    let m = Memory { y, relaxation_m };

    Ok(Step(Config::single(x), Some(m)))
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

            let r = sample_uniform(0., 1.);
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

            let r = sample_uniform(0., 1.);
            if r <= p {
                y.floor() as i32
            } else {
                y.ceil() as i32
            }
        }
    }
}
