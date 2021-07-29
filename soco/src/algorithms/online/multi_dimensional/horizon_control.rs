use crate::algorithms::online::{FractionalStep, Step};
use crate::config::{Config, FractionalConfig};
use crate::convert::ResettableProblem;
use crate::model::{ModelOutputFailure, ModelOutputSuccess};
use crate::numerics::convex_optimization::{find_minimizer, WrappedObjective};
use crate::problem::{
    FractionalSimplifiedSmoothedConvexOptimization, Online, Problem,
};
use crate::result::Result;
use crate::schedule::{FractionalSchedule, Schedule};

/// Receding Horizon Control
pub fn rhc<C, D>(
    o: Online<FractionalSimplifiedSmoothedConvexOptimization<C, D>>,
    xs: &mut FractionalSchedule,
    _: &mut Vec<()>,
    _: &(),
) -> Result<FractionalStep<()>>
where
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    let t = xs.t_end() + 1;
    let x = next(0, o, t, xs.clone());
    Ok(Step(x, None))
}

/// Averaging Fixed Horizon Control
pub fn afhc<C, D>(
    o: Online<FractionalSimplifiedSmoothedConvexOptimization<C, D>>,
    xs: &mut FractionalSchedule,
    _: &mut Vec<()>,
    _: &(),
) -> Result<FractionalStep<()>>
where
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    let t = xs.t_end() + 1;

    let mut x = Config::repeat(0., o.p.d);
    for k in 1..=o.w + 1 {
        x = x + next(k, o.clone(), t, xs.clone());
    }
    Ok(Step(x / (o.w + 1) as f64, None))
}

#[derive(Clone)]
struct ObjectiveData<'a, C, D> {
    k: i32,
    o: Online<FractionalSimplifiedSmoothedConvexOptimization<'a, C, D>>,
    t: i32,
    prev_xs: FractionalSchedule,
}

fn next<C, D>(
    k: i32,
    o: Online<FractionalSimplifiedSmoothedConvexOptimization<C, D>>,
    t: i32,
    prev_xs: FractionalSchedule,
) -> FractionalConfig
where
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
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

            p.objective_function_with_default(&xs, &prev_x)
                .unwrap()
                .cost
        },
    );

    let (raw_xs, _) = find_minimizer(objective, bounds);
    Config::new(raw_xs[0..d as usize].to_vec())
}
