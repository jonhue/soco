use nlopt::Algorithm;
use nlopt::Nlopt;
use nlopt::ObjFn;
use nlopt::Target;

use crate::problem::{ContinuousHomProblem, DiscreteHomProblem};
use crate::result::{Error, Result};
use crate::utils::{assert, is_pow_of_2};
use crate::PRECISION;

impl<'a> ContinuousHomProblem<'a> {
    /// Computes the number of servers at time `t_end` resulting in the lowest possible cost.
    pub fn find_lower_bound(&self, t_end: i32) -> Result<f64> {
        let objective_function =
            |xs: &[f64],
             _: Option<&mut [f64]>,
             p: &mut &ContinuousHomProblem<'a>|
             -> f64 { p.objective_function(&xs.to_vec()).unwrap() };

        self.find_bound(objective_function, t_end)
    }

    /// Computes the number of servers at time `t_end` resulting in the highest possible cost.
    pub fn find_upper_bound(&self, t_end: i32) -> Result<f64> {
        let objective_function = |xs: &[f64],
                                  _: Option<&mut [f64]>,
                                  p: &mut &ContinuousHomProblem<'a>|
         -> f64 {
            p.inverted_objective_function(&xs.to_vec()).unwrap()
        };

        self.find_bound(objective_function, t_end)
    }

    fn find_bound<'b>(
        &'b self,
        objective_function: impl ObjFn<&'b ContinuousHomProblem<'a>>,
        t_end: i32,
    ) -> Result<f64> {
        assert(t_end <= self.t_end, Error::LcpBoundComputationExceedsDomain)?;

        let n = self.t_end as usize - 1;
        if n == 0 {
            return Ok(0.);
        }

        let mut xs = vec![0.0; n];
        let mut opt = Nlopt::new(
            Algorithm::Bobyqa,
            n,
            objective_function,
            Target::Minimize,
            &self,
        );
        opt.set_lower_bound(0.)?;
        opt.set_upper_bound(self.m as f64)?;
        opt.set_xtol_rel(PRECISION)?;

        opt.optimize(&mut xs)?;
        Ok(xs[xs.len() - 1])
    }
}

impl<'a> DiscreteHomProblem<'a> {
    /// Computes the number of servers resulting in the lowest possible cost.
    pub fn find_lower_bound(&self) -> Result<i32> {
        let tmp;
        let p = if is_pow_of_2(self.m) {
            self
        } else {
            tmp = self.make_pow_of_2();
            &tmp
        };

        let (xs, _) = p.iopt()?;
        Ok(xs[xs.len() - 1])
    }

    /// Computes the number of servers resulting in the highest possible cost.
    pub fn find_upper_bound(&self) -> Result<i32> {
        let tmp;
        let p = if is_pow_of_2(self.m) {
            self
        } else {
            tmp = self.make_pow_of_2();
            &tmp
        };

        let (xs, _) = p.inverted_iopt()?;
        Ok(xs[xs.len() - 1])
    }
}
