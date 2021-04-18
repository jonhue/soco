#![allow(clippy::float_cmp)]

use rsdc::problem::types::{HomProblem, Online};

#[test]
fn case1() {
    let p = HomProblem {
        m: 2,
        t_end: 2,
        f: Box::new(|t, j| Some(t as f64 * (if j == 0 { 1. } else { 0. }))),
        beta: 1.,
    };
    let o = Online { p, w: 0 };
    o.verify();

    let result = o.stream(Online::lcp, |_, _, _| None);

    assert_eq!(result.0, vec![1]);
    assert_eq!(result.1, vec![(1, 1)]);
}
