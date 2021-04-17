mod analysis;
use analysis::discrete_objective_function;

mod lib;
use lib::types::HomProblem;
use lib::verifiers::{verify_discrete_problem, verify_discrete_schedule};

mod algorithms;

fn main() {
    let instance = HomProblem {
        m: 3,
        t_end: 3,
        f: Box::new(|t, _x| Some((t as f64) + 1.)),
        beta: 0.4,
    };
    verify_discrete_problem(&instance);
    let result = instance.alg1();
    verify_discrete_schedule(&instance, &result.0);
    println!("Schedule: {:?}", result.0);
    println!("Cost: {:.1}", result.1);
    println!(
        "Calculated Cost: {:.1}",
        discrete_objective_function(&instance, &result.0)
    );
}
