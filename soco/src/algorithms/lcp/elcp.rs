//! Lazy Capacity Provisioning

use crate::online::{Online, OnlineSolution};
use crate::problem::ContinuousHomProblem;
use crate::result::{Error, Result};
use crate::schedule::ContinuousSchedule;
use crate::utils::{assert, fproject};

/// Lower and upper bound at some time t.
type Memory<T> = (T, T);

impl<'a> Online<ContinuousHomProblem<'a>> {
    /// (Continuous) Extended Lazy Capacity Provisioning
    pub fn elcp(
        &self,
        xs: &ContinuousSchedule,
        _: &Vec<Memory<f64>>,
    ) -> Result<OnlineSolution<f64, Memory<f64>>> {
        assert(self.w == 0, Error::UnsupportedPredictionWindow)?;

        let i = if xs.is_empty() { 0. } else { xs[xs.len() - 1] };
        let l = self.lower_bound()?;
        let u = self.upper_bound()?;
        let j = fproject(i, l, u);
        Ok((j, (l, u)))
    }

    fn lower_bound(&self) -> Result<f64> {
        let objective_function =
            |xs: &[f64],
             _: Option<&mut [f64]>,
             p: &mut &ContinuousHomProblem<'a>|
             -> f64 { p.objective_function(&xs.to_vec()).unwrap() };

        let xs = self.past_opt(objective_function)?;
        Ok(xs[xs.len() - 1])
    }

    fn upper_bound(&self) -> Result<f64> {
        let objective_function = |xs: &[f64],
                                  _: Option<&mut [f64]>,
                                  p: &mut &ContinuousHomProblem<'a>|
         -> f64 {
            p.inverted_objective_function(&xs.to_vec()).unwrap()
        };

        let xs = self.past_opt(objective_function)?;
        Ok(xs[xs.len() - 1])
    }
}
