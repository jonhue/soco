//! Discrete Lazy Capacity Provisioning

use nlopt::Algorithm;
use nlopt::Nlopt;
use nlopt::ObjFn;
use nlopt::Target;

use crate::problem::{
    DiscreteHomProblem, DiscreteSchedule, Online, OnlineSolution,
};
use crate::utils::iproject;

/// Lower and upper bound at some time t.
type Memory = (i32, i32);

impl<'a> Online<DiscreteHomProblem<'a>> {
    pub fn lcp(
        &self,
        xs: &DiscreteSchedule,
        _: &Vec<Memory>,
    ) -> OnlineSolution<i32, Memory> {
        assert_eq!(self.w, 0);

        let i = if xs.is_empty() { 0 } else { xs[xs.len() - 1] };
        let l = self.lower_bound();
        let u = self.upper_bound();
        let j = iproject(i, l, u);
        (j, (l, u))
    }

    fn lower_bound(&self) -> i32 {
        let objective_function =
            |xs: &[f64],
             _gradient: Option<&mut [f64]>,
             p: &mut &DiscreteHomProblem<'a>|
             -> f64 { p.objective_function(&tighten(xs)) };

        let xs = self.opt(objective_function);
        xs[self.p.t_end as usize - 1].ceil() as i32
    }

    fn upper_bound(&self) -> i32 {
        let objective_function =
            |xs: &[f64],
             _gradient: Option<&mut [f64]>,
             p: &mut &DiscreteHomProblem<'a>|
             -> f64 { p.inverted_objective_function(&tighten(xs)) };

        let xs = self.opt(objective_function);
        xs[self.p.t_end as usize - 1].ceil() as i32
    }

    /// Use a relaxed convex optimization.
    fn opt<'b>(
        &'b self,
        objective_function: impl ObjFn<&'b DiscreteHomProblem<'a>>,
    ) -> Vec<f64> {
        let mut xs = vec![0.0; self.p.t_end as usize];
        let mut opt = Nlopt::new(
            Algorithm::Bobyqa,
            self.p.t_end as usize,
            objective_function,
            Target::Minimize,
            &self.p,
        );
        opt.set_lower_bound(0.).unwrap();
        opt.set_upper_bound(self.p.m as f64).unwrap();
        opt.set_xtol_abs1(1.).unwrap();

        opt.optimize(&mut xs).unwrap();
        xs
    }
}

fn tighten(xs: &[f64]) -> DiscreteSchedule {
    xs.iter().map(|&x| x.ceil() as i32).collect()
}
