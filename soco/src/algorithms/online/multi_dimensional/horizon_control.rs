use crate::algorithms::online::{FractionalStep, Step};
use crate::config::{Config, FractionalConfig};
use crate::convert::ResettableProblem;
use crate::numerics::convex_optimization::{find_minimizer, WrappedObjective};
use crate::objective::Objective;
use crate::problem::{FractionalSimplifiedSmoothedConvexOptimization, Online};
use crate::result::Result;
use crate::schedule::{FractionalSchedule, Schedule};

/// Receding Horizon Control
pub fn rhc(
    o: Online<FractionalSimplifiedSmoothedConvexOptimization>,
    xs: &mut FractionalSchedule,
    _: &mut Vec<()>,
    _: &(),
) -> Result<FractionalStep<()>> {
    let t = xs.t_end() + 1;
    let x = next(0, o, t, xs.clone())?;
    Ok(Step(x, None))
}

/// Averaging Fixed Horizon Control
pub fn afhc(
    o: Online<FractionalSimplifiedSmoothedConvexOptimization>,
    xs: &mut FractionalSchedule,
    _: &mut Vec<()>,
    _: &(),
) -> Result<FractionalStep<()>> {
    let t = xs.t_end() + 1;

    let mut x = Config::repeat(0., o.p.d);
    for k in 1..=o.w + 1 {
        x = x + next(k, o.clone(), t, xs.clone())?;
    }
    Ok(Step(x / (o.w + 1) as f64, None))
}

struct ObjectiveData<'a> {
    k: i32,
    o: Online<FractionalSimplifiedSmoothedConvexOptimization<'a>>,
    t: i32,
    prev_xs: FractionalSchedule,
}

fn next(
    k: i32,
    o: Online<FractionalSimplifiedSmoothedConvexOptimization>,
    t: i32,
    prev_xs: FractionalSchedule,
) -> Result<FractionalConfig> {
    let d = o.p.d;
    let bounds = vec![
        (0., o.p.bounds[0]);
        FractionalSchedule::raw_encoding_len(o.p.d, o.w) as usize
    ];
    let objective = WrappedObjective::new(
        ObjectiveData { k, o, t, prev_xs },
        |raw_xs, data| {
            let xs = Schedule::from_raw(data.o.p.d, data.o.w, raw_xs);
            let prev_x = if data.prev_xs.t_end() - data.k > 0 {
                data.prev_xs[(data.prev_xs.t_end() - data.k - 1) as usize]
                    .clone()
            } else {
                Config::repeat(0., data.o.p.d)
            };
            let p = data.o.p.reset(data.t - data.k);

            p.objective_function_with_default(&xs, &prev_x).unwrap()
        },
    );

    let (raw_xs, _) = find_minimizer(objective, bounds)?;
    Ok(Config::new(raw_xs[0..d as usize].to_vec()))
}
