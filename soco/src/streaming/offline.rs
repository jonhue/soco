use crate::{
    algorithms::{offline::OfflineAlgorithm, Options},
    model::{Model, OfflineInput, OnlineInput},
    problem::Problem,
    result::Result,
};

/// Generates problem instance from model and solves it using an offline algorithm.
pub fn solve<'a, T, P, O, A, B>(
    model: &'a impl Model<'a, P, A, B>,
    alg: &impl OfflineAlgorithm<T, P, O>,
    options: O,
    input: A,
    inverted: bool,
) -> Result<T>
where
    P: Problem + 'a,
    O: Options<P> + 'a,
    A: OfflineInput,
    B: OnlineInput<'a>,
{
    let p = model.to(input);
    p.verify()?;
    println!("Generated a problem instance: {:?}", p);

    println!("Simulating until time slot {}.", p.t_end());
    alg.solve(p, options, inverted)
}
