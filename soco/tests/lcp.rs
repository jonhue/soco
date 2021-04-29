#![allow(clippy::float_cmp)]

use std::sync::Arc;

use soco::problem::{HomProblem, Online};
use soco::schedule::DiscretizableSchedule;
use soco::verifiers::VerifiableSchedule;

#[test]
fn elcp1() {
    let p = HomProblem {
        m: 2,
        t_end: 1,
        f: Arc::new(|t, j| Some(t as f64 * (if j == 0. { 1. } else { 0. }))),
        beta: 1.,
    };
    let mut o = Online { p, w: 0 };
    o.verify();

    let result = o.stream(Online::elcp, |_, _, _| false);
    result.0.verify(o.p.m, o.p.t_end);

    assert_eq!(result.0.to_i(), vec![0]);
}

#[test]
fn elcp2() {
    let p = HomProblem {
        m: 2,
        t_end: 1,
        f: Arc::new(|t, j| Some(t as f64 * (if j == 0. { 1. } else { 0. }))),
        beta: 1.,
    };
    let mut o = Online { p, w: 0 };
    o.verify();

    let t_end = 2;
    let result = o.offline_stream(Online::elcp, t_end);
    result.0.verify(o.p.m, t_end);

    assert_eq!(result.0.to_i(), vec![0, 0]);
}

#[test]
fn ilcp1() {
    let p = HomProblem {
        m: 2,
        t_end: 1,
        f: Arc::new(|t, j| Some(t as f64 * (if j == 0 { 1. } else { 0. }))),
        beta: 1.,
    };
    let mut o = Online { p, w: 0 };
    o.verify();

    let result = o.stream(Online::ilcp, |_, _, _| false);
    result.0.verify(o.p.m, o.p.t_end);

    assert_eq!(result.0, vec![0]);
}

#[test]
fn ilcp2() {
    let p = HomProblem {
        m: 2,
        t_end: 1,
        f: Arc::new(|t, j| Some(t as f64 * (if j == 0 { 1. } else { 0. }))),
        beta: 1.,
    };
    let mut o = Online { p, w: 0 };
    o.verify();

    let t_end = 2;
    let result = o.offline_stream(Online::ilcp, t_end);
    result.0.verify(o.p.m, t_end);

    assert_eq!(result.0, vec![0, 0]);
}
