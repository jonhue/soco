use crate::{
    algorithms::{
        offline::{OfflineAlgorithm, OfflineResult},
        Options,
    },
    model::{Model, OfflineInput, OnlineInput},
    objective::Objective,
    problem::Problem,
    result::Result,
    schedule::Schedule,
    value::Value,
};

/// Generates problem instance from model and solves it using an offline algorithm.
pub fn solve<'a, T, R, P, O, A, B>(
    model: &'a impl Model<P, A, B>,
    alg: &impl OfflineAlgorithm<T, R, P, O>,
    options: O,
    input: A,
    inverted: bool,
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
    println!("Generated a problem instance: {:?}", p);

    println!("Simulating until time slot {}.", p.t_end());
    let result = alg.solve(p.clone(), options, inverted)?;

    let xs = result.xs();
    let cost = p.objective_function(&xs)?;
    Ok((xs, cost))
}
