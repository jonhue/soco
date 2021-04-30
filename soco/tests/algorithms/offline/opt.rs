#![allow(clippy::float_cmp)]

// use rand::prelude::*;
// use rand_pcg::Pcg64;
use std::sync::Arc;

use soco::convert::DiscretizableSchedule;
use soco::problem::HomProblem;
use soco::verifiers::{VerifiableProblem, VerifiableSchedule};

#[test]
fn opt_1() {
    let p = HomProblem {
        m: 2,
        t_end: 2,
        f: Arc::new(|t, j| Some(t as f64 * (if j == 0. { 1. } else { 0. }))),
        beta: 1.,
    };
    p.verify().unwrap();

    let result = p.opt().unwrap();
    result.0.verify(p.m, p.t_end).unwrap();

    assert_eq!(result.0.to_i(), vec![1, 1]);
    assert_eq!(result.1, p.objective_function(&result.0).unwrap());
}

// #[test]
// fn opt_2() {
//     let p = HomProblem {
//         m: 8,
//         t_end: 50,
//         f: Arc::new(|t, j| {
//             Some(
//                 Pcg64::seed_from_u64((t as f64 * j) as u64).gen_range(0.0..1_000_000.),
//             )
//         }),
//         beta: 1.,
//     };
//     p.verify().unwrap();

//     let result = p.opt().unwrap();
//     result.0.verify(p.m, p.t_end).unwrap();

//     assert_eq!(result.1, p.objective_function(&result.0).unwrap());
// }
