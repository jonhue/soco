//! Algorithms by Bansal et al.

use nlopt::Algorithm;
use nlopt::Nlopt;
use nlopt::ObjFn;
use nlopt::Target;

use crate::problem::{
    ContinuousHomProblem, ContinuousSchedule, Online, OnlineSolution,
};
use crate::PRECISION;

pub type Memory = ();

impl<'a> Online<ContinuousHomProblem<'a>> {
    /// Deterministic Online Algorithm
    pub fn bansal(
        &self,
        _: &ContinuousSchedule,
        _: &Vec<Memory>,
    ) -> OnlineSolution<f64, Memory> {
        (0., ())
    }

    /// Memoryless Deterministic Online Algorithm
    pub fn mbansal(
        &self,
        xs: &ContinuousSchedule,
        _: &Vec<()>,
    ) -> OnlineSolution<f64, ()> {
        let prev_x = if xs.is_empty() { 0. } else { xs[xs.len() - 1] };
        let f = |j: f64| (self.p.f)(xs.len() as i32 + 1, j).unwrap();
        let constraint =
            |xs: &[f64], _: Option<&mut [f64]>, _: &mut ()| -> f64 {
                (xs[0] - prev_x).abs() - f(xs[0]) / 2.
            };
        let x = argmin(f, 0., self.p.m as f64, vec![constraint]);

        (x, ())
    }
}

/// Determines the minimizer of a convex function `f` on the interval `[a, b]` subject to some inequality constraints (`f <= 0`).
fn argmin(
    f: impl Fn(f64) -> f64,
    a: f64,
    b: f64,
    inequality_constraints: Vec<impl ObjFn<()>>,
) -> f64 {
    let objective_function =
        |xs: &[f64], _: Option<&mut [f64]>, _: &mut ()| -> f64 { f(xs[0]) };

    let mut xs = [0.0];
    let mut opt = Nlopt::new(
        Algorithm::Bobyqa,
        1,
        objective_function,
        Target::Minimize,
        (),
    );
    opt.set_lower_bound(a).unwrap();
    opt.set_upper_bound(b).unwrap();
    opt.set_xtol_rel(PRECISION).unwrap();

    for constraint in inequality_constraints {
        opt.add_inequality_constraint(constraint, (), PRECISION)
            .unwrap();
    }

    opt.optimize(&mut xs).unwrap();
    xs[0]
}
