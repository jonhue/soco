use crate::online::{Online, OnlineSolution};
use crate::problem::ContinuousHomProblem;
use crate::result::{Error, Result};
use crate::schedule::ContinuousSchedule;
use crate::utils::{assert, fproject};

/// Lower and upper bound at some time t.
type Memory<T> = (T, T);

impl<'a> Online<ContinuousHomProblem<'a>> {
    /// (Continuous) Lazy Capacity Provisioning
    pub fn lcp(
        &self,
        xs: &ContinuousSchedule,
        _: &Vec<Memory<f64>>,
    ) -> Result<OnlineSolution<f64, Memory<f64>>> {
        assert(self.w == 0, Error::UnsupportedPredictionWindow)?;

        let i = if xs.is_empty() { 0. } else { xs[xs.len() - 1] };
        let l = self.p.find_lower_bound(self.p.t_end)?;
        let u = self.p.find_upper_bound(self.p.t_end)?;
        let j = fproject(i, l, u);
        Ok((j, (l, u)))
    }
}
