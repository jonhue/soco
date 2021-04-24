#![allow(clippy::float_cmp)]

use rsdc::problem::{HomProblem, Online};
use rsdc::schedule::DiscretizableSchedule;
use rsdc::verifiers::{VerifiableSchedule};

#[test]
fn lcp1() {
    let p = HomProblem {
        m: 2,
        t_end: 1,
        f: Box::new(|t, j: f64| Some(t as f64 * (if j == 0. { 1. } else { 0. }))),
        beta: 1.,
    };
    let o = Online { p, w: 0 };
    o.verify();

    let result = o.stream(Online::lcp, |_, _, _| None);
    result.0.verify(&o.p);

    assert_eq!(result.0.to_i(), vec![1]);
}

#[test]
fn ilcp1() {
    let p = HomProblem {
        m: 2,
        t_end: 1,
        f: Box::new(|t, j| Some(t as f64 * (if j == 0 { 1. } else { 0. }))),
        beta: 1.,
    };
    let o = Online { p, w: 0 };
    o.verify();

    let result = o.stream(Online::ilcp, |_, _, _| None);
    result.0.verify(&o.p);

    assert_eq!(result.0, vec![1]);
}
