use bacon_sci::differentiate::second_derivative;
use bacon_sci::integrate::integrate;
use nlopt::Algorithm;
use nlopt::Nlopt;
use nlopt::Target;
use std::f64::INFINITY;

use crate::problem::{
    ContinuousHomProblem, ContinuousSchedule, Online, OnlineSolution,
};
use crate::PRECISION;

/// Probability distribution over the number of servers.
pub type Memory<'a> = Box<dyn Fn(f64) -> f64 + 'a>;

static STEP_SIZE: f64 = 1e-16;

impl<'a> Online<ContinuousHomProblem<'a>> {
    /// Randomized Online Algorithm
    pub fn rand(
        &self,
        xs: &ContinuousSchedule,
        ps: &Vec<Memory<'a>>,
    ) -> OnlineSolution<f64, Memory<'a>> {
        let t = xs.len() as i32 + 1;
        let prev_p = if ps.is_empty() {
            Box::new(|j: f64| if j == 0. { 1. } else { 0. })
        } else {
            ps[ps.len() - 1]
        };

        let x_m = self.find_minimizer(t);
        let x_r = self.find_right_bound(t, prev_p, x_m);

        // (1., prev_p)
    }

    /// Determines minimizer of `f` with a convex optimization.
    fn find_minimizer(&self, t: i32) -> f64 {
        let objective_function =
            |xs: &[f64], _: Option<&mut [f64]>, _: &mut ()| -> f64 {
                (self.p.f)(t, xs[0]).unwrap()
            };
        let mut xs = [0.0];
        let mut opt = Nlopt::new(
            Algorithm::Bobyqa,
            1,
            objective_function,
            Target::Minimize,
            (),
        );
        opt.set_lower_bound(0.).unwrap();
        opt.set_upper_bound(self.p.m as f64).unwrap();
        opt.set_xtol_rel(PRECISION).unwrap();
        opt.optimize(&mut xs).unwrap();
        xs[0]
    }

    /// Determines `x_r` with a convex optimization.
    fn find_right_bound(&self, t: i32, prev_p: Memory<'a>, x_m: f64) -> f64 {
        let objective_function =
            |xs: &[f64], _: Option<&mut [f64]>, _: &mut ()| -> f64 { xs[0] };
        let mut xs = [x_m];
        let mut opt = Nlopt::new(
            Algorithm::Bobyqa,
            1,
            objective_function,
            Target::Maximize,
            (),
        );
        opt.set_lower_bound(0.).unwrap();
        opt.set_upper_bound(self.p.m as f64).unwrap();
        opt.set_xtol_rel(PRECISION).unwrap();
        opt.add_equality_constraint(
            |xs: &[f64], _: Option<&mut [f64]>, _: &mut ()| -> f64 {
                integrate(
                    x_m,
                    xs[0],
                    |j: f64| {
                        second_derivative(
                            |j: f64| (self.p.f)(t, j).unwrap(),
                            j,
                            STEP_SIZE,
                        )
                    },
                    PRECISION,
                )
                .unwrap()
                    - integrate(xs[0], INFINITY, |j: f64| prev_p(j), PRECISION)
                        .unwrap()
            },
            (),
            PRECISION,
        )
        .unwrap();
        opt.optimize(&mut xs).unwrap();
        xs[0]
    }
}
