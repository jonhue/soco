use crate::PRECISION;
use crate::problem::ContinuousHomProblem;
use crate::result::{Error, Result};
use crate::schedule::ContinuousSchedule;
use crate::utils::{assert, fproject};

impl<'a> ContinuousHomProblem<'a> {
    /// Deterministic Offline Algorithm
    pub fn opt(&self) -> Result<(ContinuousSchedule, f64)> {
        let mut xs = Vec::new();

        let mut x = 0.;
        let mut cost = 0.;
        for t in (1..=self.t_end).rev() {
            let l = self.find_lower_bound(t)?;
            let u = self.find_upper_bound(t)?;
            if t == self.t_end {
                assert((l - u).abs() < PRECISION, Error::LcpBoundMismatch)?;
                cost = l;
            };

            x = fproject(x, l, u);
            xs.insert(0, x);
        }

        Ok((xs, cost))
    }
}
