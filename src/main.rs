mod lib;
use lib::analysis::discrete_objective_function;
use lib::types::{Problem, Schedule};
use lib::verifiers::{verify_discrete_problem, verify_discrete_schedule};

mod algorithms;
use algorithms::alg1;

fn f1(t: usize, x_t: i32) -> Option<f64> {
    return Some((t as f64) + 1.);
}

fn main() {
    println!("Hello, world!");

    let instance = Problem {
        m: 3,
        t_end: 3,
        f: f1,
        beta: 0.4,
    };
    verify_discrete_problem(&instance);
    let schedule: Schedule<i32> = vec![2, 3, 1];
    verify_discrete_schedule(&instance, &schedule);
    println!(
        "Cost: {:.1}",
        discrete_objective_function(&instance, &schedule)
    )
}
