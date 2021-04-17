use crate::lib::analysis::ObjectiveFunction;
use crate::lib::types::HomProblem;
use crate::lib::verifiers::{VerifiableProblem, VerifiableSchedule};
use ordered_float::OrderedFloat;

#[test]
fn case1() {
    let p = HomProblem {
        m: 3,
        t_end: 3,
        f: Box::new(|t, _x| Some((t as f64) + 1.)),
        beta: 0.4,
    };
    p.verify();

    let result = p.alg1();
    result.0.verify(&p);

    assert_eq!(result.1, p.objective_function(&result.0));
    assert_eq!(result, (vec![], OrderedFloat(0.)));
}
