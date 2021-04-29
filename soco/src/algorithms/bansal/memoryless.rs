use nlopt::Algorithm;
use nlopt::Nlopt;
use nlopt::Target;

use crate::problem::{
    ContinuousHomProblem, ContinuousSchedule, Online, OnlineSolution,
};
use crate::PRECISION;

impl<'a> Online<ContinuousHomProblem<'a>> {
    /// Memoryless Deterministic Online Algorithm
    pub fn memoryless(
        &self,
        xs: &ContinuousSchedule,
        _: &Vec<()>,
    ) -> OnlineSolution<f64, ()> {
        let t = xs.len() as i32 + 1;
        let prev_x = if xs.is_empty() { 0. } else { xs[xs.len() - 1] };

        let x = self.next(t, prev_x);
        (x, ())
    }

    /// Determines next `x` with a convex optimization.
    fn next(&self, t: i32, prev_x: f64) -> f64 {
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
        opt.add_inequality_constraint(
            |xs: &[f64], _: Option<&mut [f64]>, _: &mut ()| -> f64 {
                (xs[0] - prev_x).abs() - (self.p.f)(t, xs[0]).unwrap() / 2.
            },
            (),
            PRECISION,
        )
        .unwrap();
        opt.optimize(&mut xs).unwrap();
        xs[0]
    }
}
