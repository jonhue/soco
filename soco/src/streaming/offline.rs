use crate::{
    algorithms::{
        offline::{OfflineAlgorithm, OfflineOptions, OfflineResult},
        Options,
    },
    model::{Model, OfflineInput, OnlineInput},
    objective::Objective,
    problem::Problem,
    result::Result,
    schedule::Schedule,
    value::Value,
};
use log::info;

/// Generates problem instance from model and solves it using an offline algorithm.
pub fn solve<'a, T, R, P, O, A, B>(
    model: &'a impl Model<P, A, B>,
    alg: &impl OfflineAlgorithm<T, R, P, O>,
    options: O,
    offline_options: OfflineOptions,
    input: A,
) -> Result<(Schedule<T>, f64)>
where
    T: Value<'a>,
    R: OfflineResult<T>,
    P: Objective<'a, T> + Problem + 'a,
    O: Options<P> + 'a,
    A: OfflineInput,
    B: OnlineInput,
{
    let p = model.to(input);
    p.verify()?;
    info!("Generated a problem instance: {:?}", p);

    info!("Simulating until time slot {}.", p.t_end());
    let result = alg.solve(p.clone(), options, offline_options)?;

    let xs = result.xs();
    let cost = p.objective_function(&xs)?;
    Ok((xs, cost.raw()))
}
