use nlopt::Algorithm;
use nlopt::Nlopt;
use nlopt::ObjFn;
use nlopt::Target;

use crate::algorithms::offline::iopt::{inverted_iopt, iopt, make_pow_of_2};
use crate::objective::Objective;
use crate::problem::{ContinuousHomProblem, DiscreteHomProblem};
use crate::result::{Error, Result};
use crate::utils::{assert, is_pow_of_2};
use crate::PRECISION;

pub trait Bounded<T> {
    /// Computes the number of servers at time `t_end` resulting in the lowest possible cost.
    fn find_lower_bound(&self, t_end: i32) -> Result<T>;

    /// Computes the number of servers at time `t_end` resulting in the highest possible cost.
    fn find_upper_bound(&self, t_end: i32) -> Result<T>;
}

impl Bounded<f64> for ContinuousHomProblem<'_> {
    fn find_lower_bound(&self, t_end: i32) -> Result<f64> {
        let objective_function =
            |xs: &[f64],
             _: Option<&mut [f64]>,
             p: &mut &ContinuousHomProblem<'_>|
             -> f64 { p.objective_function(&xs.to_vec()).unwrap() };

        find_bound(self, objective_function, t_end)
    }

    fn find_upper_bound(&self, t_end: i32) -> Result<f64> {
        let objective_function = |xs: &[f64],
                                  _: Option<&mut [f64]>,
                                  p: &mut &ContinuousHomProblem<'_>|
         -> f64 {
            p.inverted_objective_function(&xs.to_vec()).unwrap()
        };

        find_bound(self, objective_function, t_end)
    }
}

fn find_bound<'a>(
    p: &'a ContinuousHomProblem<'a>,
    objective_function: impl ObjFn<&'a ContinuousHomProblem<'a>>,
    t_end: i32,
) -> Result<f64> {
    assert(t_end <= p.t_end, Error::LcpBoundComputationExceedsDomain)?;

    let n = p.t_end as usize - 1;
    if n == 0 {
        return Ok(0.);
    }

    let mut xs = vec![0.0; n];
    let mut opt = Nlopt::new(
        Algorithm::Bobyqa,
        n,
        objective_function,
        Target::Minimize,
        p,
    );
    opt.set_lower_bound(0.)?;
    opt.set_upper_bound(p.m as f64)?;
    opt.set_xtol_rel(PRECISION)?;

    opt.optimize(&mut xs)?;
    Ok(xs[xs.len() - 1])
}

impl Bounded<i32> for DiscreteHomProblem<'_> {
    fn find_lower_bound(&self, t_end: i32) -> Result<i32> {
        assert(t_end == self.t_end, Error::UnsupportedBoundsCalculation)?;

        let tmp;
        let p = if is_pow_of_2(self.m) {
            self
        } else {
            tmp = make_pow_of_2(self);
            &tmp
        };

        let (xs, _) = iopt(p)?;
        Ok(xs[xs.len() - 1])
    }

    fn find_upper_bound(&self, t_end: i32) -> Result<i32> {
        assert(t_end == self.t_end, Error::UnsupportedBoundsCalculation)?;

        let tmp;
        let p = if is_pow_of_2(self.m) {
            self
        } else {
            tmp = make_pow_of_2(self);
            &tmp
        };

        let (xs, _) = inverted_iopt(p)?;
        Ok(xs[xs.len() - 1])
    }
}
