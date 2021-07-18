use crate::algorithms::online::{FractionalStep, Step};
use crate::config::{Config, FractionalConfig};
use crate::convert::ResettableProblem;
use crate::numerics::convex_optimization::find_minimizer;
use crate::objective::Objective;
use crate::problem::{FractionalSimplifiedSmoothedConvexOptimization, Online};
use crate::result::Result;
use crate::schedule::{FractionalSchedule, Schedule};
use noisy_float::prelude::*;

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
    let bounds = vec![
        (0., o.p.bounds[0]);
        FractionalSchedule::raw_encoding_len(o.p.d, o.w) as usize
    ];
    let objective = |raw_xs: &[f64]| -> R64 {
        let xs = Schedule::from_raw(o.p.d, o.w, raw_xs);
        let prev_x = if prev_xs.t_end() - k > 0 {
            prev_xs[(prev_xs.t_end() - k - 1) as usize].clone()
        } else {
            Config::repeat(0., o.p.d)
        };
        let p = o.p.reset(t - k);

        r64(p
            .objective_function_with_default(&xs, &prev_x, false)
            .unwrap())
    };

    let (raw_xs, _) = find_minimizer(objective, &bounds)?;
    Ok(Config::new(raw_xs[0..o.p.d as usize].to_vec()))
}
