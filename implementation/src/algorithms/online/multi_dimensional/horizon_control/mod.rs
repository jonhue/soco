use crate::config::{Config, FractionalConfig};
use crate::convert::ResettableProblem;
use crate::model::{ModelOutputFailure, ModelOutputSuccess};
use crate::numerics::convex_optimization::{find_minimizer, WrappedObjective};
use crate::problem::{
    FractionalSimplifiedSmoothedConvexOptimization, Online, Problem,
};
use crate::schedule::Schedule;

pub mod averaging_fixed_horizon_control;
pub mod receding_horizon_control;

#[derive(Clone)]
struct ObjectiveData<'a, C, D> {
    t_start: i32,
    o: Online<FractionalSimplifiedSmoothedConvexOptimization<'a, C, D>>,
    prev_x: FractionalConfig,
}

/// Returns new initial config $prev_x$ and config for time slot $t$.
fn next<C, D>(
    k: i32,
    o: Online<FractionalSimplifiedSmoothedConvexOptimization<C, D>>,
    t: i32,
    prev_x: FractionalConfig,
) -> (FractionalConfig, FractionalConfig)
where
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    assert!(1 <= k && k <= o.w + 1);
    let t_start = t + k - (o.w + 1);

    let d = o.p.d;
    let bounds = (0..o.w + 1)
        .into_iter()
        .map(|_| (0..o.p.d as usize).into_iter().map(|k| (0., o.p.bounds[k])))
        .flatten()
        .collect();
    let objective = WrappedObjective::new(
        ObjectiveData {
            t_start,
            o,
            prev_x,
        },
        |raw_xs, data| {
            let xs = Schedule::from_raw(data.o.p.d, data.o.w + 1, raw_xs);
            let p = data.o.p.reset(data.t_start - 1);
            p.objective_function_with_default(&xs, &data.prev_x)
                .unwrap()
                .cost
        },
    );

    let (raw_xs, _) = find_minimizer(objective, bounds);
    let offset = d * (t - t_start);
    (
        Config::new(raw_xs[0..d as usize].to_vec()),
        Config::new(
            raw_xs[offset as usize..offset as usize + d as usize].to_vec(),
        ),
    )
}
