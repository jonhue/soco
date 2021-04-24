#![allow(clippy::float_cmp)]

use ordered_float::OrderedFloat;
use rand::prelude::*;
use rand_pcg::Pcg64;

use rsdc::problem::HomProblem;
use rsdc::verifiers::{VerifiableProblem, VerifiableSchedule};

#[test]
fn transform1() {
    let p = HomProblem {
        m: 103,
        t_end: 1_000,
        f: Box::new(|_, _| Some(1.)),
        beta: 1.,
    };
    p.verify();
    let transformed_p = p.transform();
    transformed_p.verify();

    assert_eq!(transformed_p.m, 128);
    assert_eq!(transformed_p.t_end, p.t_end);
    assert_eq!(transformed_p.beta, p.beta);

    for t in 1..=transformed_p.t_end {
        for j in 0..=transformed_p.m {
            assert_eq!(
                (transformed_p.f)(t, j).expect(""),
                if j <= p.m { 1. } else { j as f64 * 2. },
                "f is wrongly defined for t={}, j={}",
                t,
                j
            );
        }
    }
}

#[test]
fn iopt1() {
    let p = HomProblem {
        m: 2,
        t_end: 2,
        f: Box::new(|t, j| Some(t as f64 * (if j == 0 { 1. } else { 0. }))),
        beta: 1.,
    };
    p.verify();

    let result = p.iopt();
    result.0.verify(&p);

    assert_eq!(result, (vec![1, 1], OrderedFloat(1.)));
    assert_eq!(result.1, p.objective_function(&result.0));
}

#[test]
fn iopt2() {
    let p = HomProblem {
        m: 8,
        t_end: 100,
        f: Box::new(|t, j| {
            Some(
                Pcg64::seed_from_u64((t * j) as u64).gen_range(0.0..1_000_000.),
            )
        }),
        beta: 1.,
    };
    p.verify();

    let result = p.iopt();
    result.0.verify(&p);

    assert_eq!(result.1, p.objective_function(&result.0));
}

#[test]
fn iopt3() {
    let p = HomProblem {
        m: 9,
        t_end: 1_000,
        f: Box::new(|t, j| {
            Some(
                Pcg64::seed_from_u64((t * j) as u64).gen_range(0.0..1_000_000.),
            )
        }),
        beta: 1.,
    };
    p.verify();

    let transformed_p = p.transform();
    let result = transformed_p.iopt();
    result.0.verify(&transformed_p);

    assert_eq!(result.1, transformed_p.objective_function(&result.0));
}
