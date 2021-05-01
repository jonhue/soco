use crate::online::{Online, OnlineSolution};
use crate::problem::DiscreteHomProblem;
use crate::result::{Error, Result};
use crate::schedule::DiscreteSchedule;
use crate::utils::{assert, project};

/// Lower and upper bound at some time t.
type Memory<T> = (T, T);

impl<'a> Online<DiscreteHomProblem<'a>> {
    /// Integer Lazy Capacity Provisioning
    pub fn ilcp(
        &self,
        xs: &DiscreteSchedule,
        _: &Vec<Memory<i32>>,
    ) -> Result<OnlineSolution<i32, Memory<i32>>> {
        assert(self.w == 0, Error::UnsupportedPredictionWindow)?;

        let i = if xs.is_empty() { 0 } else { xs[xs.len() - 1] };
        let l = self.p.find_lower_bound()?;
        let u = self.p.find_upper_bound()?;
        let j = project(i, l, u);
        Ok((j, (l, u)))
    }
}
