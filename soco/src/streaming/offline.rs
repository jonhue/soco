use std::{panic, time::Instant};

use crate::{
    algorithms::{
        offline::{OfflineAlgorithm, OfflineOptions, OfflineResult},
        Options,
    },
    cost::Cost,
    model::{
        Model, ModelOutputFailure, ModelOutputSuccess, OfflineInput,
        OnlineInput,
    },
    problem::Problem,
    result::Result,
    schedule::Schedule,
    value::Value,
};
use backtrace::Backtrace;
use log::{info, warn};

/// Generates problem instance from model and solves it using an offline algorithm.
pub fn solve<'a, T, R, P, O, A, B, C, D>(
    model: &'a impl Model<T, P, A, B, C, D>,
    alg: &impl OfflineAlgorithm<T, R, P, O, C, D>,
    options: O,
    offline_options: OfflineOptions,
    input: A,
) -> Result<(Schedule<T>, Cost<C, D>, u128)>
where
    T: Value<'a>,
    R: OfflineResult<T>,
    P: Problem<T, C, D> + 'a,
    O: Options<T, P, C, D> + 'a,
    A: OfflineInput,
    B: OnlineInput,
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    panic::set_hook(Box::new(|_panic_info| {
        warn!("\n\n{:?}", Backtrace::new());
    }));

    let p = model.to(input);
    info!("Generated a problem instance: {:?}", p);
    p.verify()?;

    info!("Simulating until time slot {}.", p.t_end());
    let start = Instant::now();
    let result = alg.solve(p.clone(), options, offline_options)?;
    let runtime = start.elapsed().as_millis();

    let xs = result.xs();
    let cost = p.objective_function(&xs)?;
    info!("Completed with {:?} and {:?}", cost, xs);
    Ok((xs, cost, runtime))
}
