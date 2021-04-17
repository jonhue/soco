mod algorithms;

mod lib;
use lib::analysis::ObjectiveFunction;
use lib::types::HomProblem;
use lib::verifiers::{VerifiableProblem, VerifiableSchedule};

fn main() {
    let instance = HomProblem {
        m: 3,
        t_end: 3,
        f: Box::new(|t, _x| Some((t as f64) + 1.)),
        beta: 0.4,
    };
    instance.verify();
    let result = instance.alg1();
    result.0.verify(&instance);
    println!("Schedule: {:?}", result.0);
    println!("Cost: {:.1}", result.1);
    println!(
        "Calculated Cost: {:.1}",
        instance.objective_function(&result.0)
    );
}
