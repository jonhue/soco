use nlopt::Algorithm;
use nlopt::Nlopt;
use nlopt::Target;

use crate::online::{Online, OnlineSolution};
use crate::problem::ContinuousHomProblem;
use crate::result::{Error, Result};
use crate::schedule::ContinuousSchedule;
use crate::utils::assert;
use crate::PRECISION;

/// Memoryless Deterministic Online Algorithm
pub fn memoryless(
    o: &Online<ContinuousHomProblem<'_>>,
    xs: &ContinuousSchedule,
    _: &Vec<()>,
) -> Result<OnlineSolution<f64, ()>> {
    assert(o.w == 0, Error::UnsupportedPredictionWindow)?;

    let t = xs.len() as i32 + 1;
    let prev_x = if xs.is_empty() { 0. } else { xs[xs.len() - 1] };

    let x = next(o, t, prev_x)?;
    Ok((x, ()))
}

/// Determines next `x` with a convex optimization.
fn next(
    o: &Online<ContinuousHomProblem<'_>>,
    t: i32,
    prev_x: f64,
) -> Result<f64> {
    let objective_function = |xs: &[f64],
                              _: Option<&mut [f64]>,
                              _: &mut ()|
     -> f64 { (o.p.f)(t, xs[0]).unwrap() };
    let mut xs = [0.0];
    let mut opt = Nlopt::new(
        Algorithm::Bobyqa,
        1,
        objective_function,
        Target::Minimize,
        (),
    );
    opt.set_lower_bound(0.)?;
    opt.set_upper_bound(o.p.m as f64)?;
    opt.set_xtol_rel(PRECISION)?;
    opt.add_inequality_constraint(
        |xs: &[f64], _: Option<&mut [f64]>, _: &mut ()| -> f64 {
            (xs[0] - prev_x).abs() - (o.p.f)(t, xs[0]).unwrap() / 2.
        },
        (),
        PRECISION,
    )?;
    opt.optimize(&mut xs)?;
    Ok(xs[0])
}
