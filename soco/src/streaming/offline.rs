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
    objective::Objective,
    problem::Problem,
    result::Result,
    schedule::Schedule,
    value::Value,
};
use log::info;

/// Generates problem instance from model and solves it using an offline algorithm.
pub fn solve<'a, T, R, P, O, A, B, C, D>(
    model: &'a impl Model<T, P, A, B, C, D>,
    alg: &impl OfflineAlgorithm<T, R, P, O, C, D>,
    options: O,
    offline_options: OfflineOptions,
    input: A,
) -> Result<(Schedule<T>, Cost<C, D>)>
where
    T: Value<'a>,
    R: OfflineResult<T>,
    P: Objective<'a, T, C, D> + Problem<T, C, D> + 'a,
    O: Options<T, P, C, D> + 'a,
    A: OfflineInput,
    B: OnlineInput,
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    let p = model.to(input);
    p.verify()?;
    info!("Generated a problem instance: {:?}", p);

    info!("Simulating until time slot {}.", p.t_end());
    let result = alg.solve(p.clone(), options, offline_options)?;

    let xs = result.xs();
    let cost = p.objective_function(&xs)?;
    Ok((xs, cost))
}
