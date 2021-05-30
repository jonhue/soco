use nlopt::{Algorithm, Nlopt, Target};

use crate::config::{Config, FractionalConfig};
use crate::convert::ResettableProblem;
use crate::objective::Objective;
use crate::online::{FractionalStep, Online, Step};
use crate::problem::FractionalSimplifiedSmoothedConvexOptimization;
use crate::result::Result;
use crate::schedule::{FractionalSchedule, Schedule};
use crate::PRECISION;

/// Receding Horizon Control
pub fn rhc(
    o: &Online<FractionalSimplifiedSmoothedConvexOptimization>,
    xs: &mut FractionalSchedule,
    _: &mut Vec<()>,
    _: &(),
) -> Result<FractionalStep<()>> {
    let t = xs.t_end() + 1;
    let x = next(0, o, t, xs)?;
    Ok(Step(x, None))
}

/// Averaging Fixed Horizon Control
pub fn afhc(
    o: &Online<FractionalSimplifiedSmoothedConvexOptimization>,
    xs: &mut FractionalSchedule,
    _: &mut Vec<()>,
    _: &(),
) -> Result<FractionalStep<()>> {
    let t = xs.t_end() + 1;

    let mut x = Config::repeat(0., o.p.d);
    for k in 1..=o.w + 1 {
        x = x + next(k, o, t, xs)?;
    }
    Ok(Step(x / (o.w + 1) as f64, None))
}

fn next(
    k: i32,
    o: &Online<FractionalSimplifiedSmoothedConvexOptimization>,
    t: i32,
    prev_xs: &FractionalSchedule,
) -> Result<FractionalConfig> {
    let d = (o.p.d * (o.w + 1)) as usize;
    let mut xs = vec![0.; d];

    let objective_function =
        |xs: &[f64], _: Option<&mut [f64]>, _: &mut ()| -> f64 {
            let prev_x = if prev_xs.t_end() - k > 0 {
                prev_xs[(prev_xs.t_end() - k - 1) as usize].clone()
            } else {
                Config::repeat(0., o.p.d)
            };
            let p = o.p.reset(t - k);

            p.objective_function_with_default(
                &build_schedule(o.p.d, o.w, xs),
                &prev_x,
                false,
            )
            .unwrap()
        };

    let mut opt = Nlopt::new(
        Algorithm::Bobyqa,
        d,
        objective_function,
        Target::Minimize,
        (),
    );
    opt.set_lower_bound(0.)?;
    opt.set_upper_bound(o.p.bounds[0])?;
    opt.set_xtol_rel(PRECISION)?;

    opt.optimize(&mut xs)?;
    Ok(Config::new(xs[0..o.p.d as usize].to_vec()))
}

fn build_schedule(d: i32, w: i32, raw_xs: &[f64]) -> FractionalSchedule {
    let mut xs = Schedule::empty();
    for t in 0..=w as usize {
        let i = d as usize * t;
        let x = Config::new(raw_xs[i..i + d as usize].to_vec());
        xs.push(x);
    }
    xs
}
