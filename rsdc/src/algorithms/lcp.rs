//! Lazy Capacity Provisioning

use nlopt::Algorithm;
use nlopt::Nlopt;
use nlopt::ObjFn;
use nlopt::Target;

use crate::problem::{
    ContinuousHomProblem, ContinuousSchedule, DiscreteHomProblem,
    DiscreteSchedule, HomProblem, Online, OnlineSolution,
};
use crate::schedule::DiscretizableSchedule;
use crate::utils::{fproject, iproject, to_vec};

/// Lower and upper bound at some time t.
type Memory<T> = (T, T);

impl<'a, T> Online<HomProblem<'a, T>> {
    /// Convex (continuous) cost optimization.
    fn past_opt<'b>(
        &'b self,
        objective_function: impl ObjFn<&'b HomProblem<'a, T>>,
    ) -> Vec<f64> {
        let n = self.p.t_end as usize - 1;
        if n == 0 {
            return vec![0.];
        }

        let mut xs = vec![0.0; n];
        let mut opt = Nlopt::new(
            Algorithm::Bobyqa,
            n,
            objective_function,
            Target::Minimize,
            &self.p,
        );
        opt.set_lower_bound(0.).unwrap();
        opt.set_upper_bound(self.p.m as f64).unwrap();
        opt.set_xtol_rel(1e-6).unwrap();

        opt.optimize(&mut xs).unwrap();
        xs
    }
}

impl<'a> Online<ContinuousHomProblem<'a>> {
    /// (Continuous) Lazy Capacity Provisioning
    pub fn lcp(
        &self,
        xs: &ContinuousSchedule,
        _: &Vec<Memory<f64>>,
    ) -> OnlineSolution<f64, Memory<f64>> {
        assert_eq!(self.w, 0);

        let i = if xs.is_empty() { 0. } else { xs[xs.len() - 1] };
        let l = self.lower_bound();
        let u = self.upper_bound();
        let j = fproject(i, l, u);
        (j, (l, u))
    }

    fn lower_bound(&self) -> f64 {
        let objective_function =
            |xs: &[f64],
             _gradient: Option<&mut [f64]>,
             p: &mut &ContinuousHomProblem<'a>|
             -> f64 { p.objective_function(&to_vec(xs)) };

        let xs = self.past_opt(objective_function);
        xs[xs.len() - 1]
    }

    fn upper_bound(&self) -> f64 {
        let objective_function =
            |xs: &[f64],
             _gradient: Option<&mut [f64]>,
             p: &mut &ContinuousHomProblem<'a>|
             -> f64 { p.inverted_objective_function(&to_vec(xs)) };

        let xs = self.past_opt(objective_function);
        xs[xs.len() - 1]
    }
}

impl<'a> Online<DiscreteHomProblem<'a>> {
    /// Integer Lazy Capacity Provisioning
    pub fn ilcp(
        &self,
        xs: &DiscreteSchedule,
        _: &Vec<Memory<i32>>,
    ) -> OnlineSolution<i32, Memory<i32>> {
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
             -> f64 { p.objective_function(&to_vec(xs).to_i()) };

        let xs = self.past_opt(objective_function);
        xs[xs.len() - 1].ceil() as i32
    }

    fn upper_bound(&self) -> i32 {
        let objective_function = |xs: &[f64],
                                  _gradient: Option<&mut [f64]>,
                                  p: &mut &DiscreteHomProblem<'a>|
         -> f64 {
            p.inverted_objective_function(&to_vec(xs).to_i())
        };

        let xs = self.past_opt(objective_function);
        xs[xs.len() - 1].ceil() as i32
    }
}
