//! Lazy Capacity Provisioning

use crate::convert::DiscretizableSchedule;
use crate::online::{Online, OnlineSolution};
use crate::problem::DiscreteHomProblem;
use crate::result::Result;
use crate::schedule::DiscreteSchedule;
use crate::utils::iproject;

/// Lower and upper bound at some time t.
type Memory<T> = (T, T);

impl<'a> Online<DiscreteHomProblem<'a>> {
    /// Integer Lazy Capacity Provisioning
    pub fn ilcp(
        &self,
        xs: &DiscreteSchedule,
        _: &Vec<Memory<i32>>,
    ) -> Result<OnlineSolution<i32, Memory<i32>>> {
        assert_eq!(self.w, 0);

        let i = if xs.is_empty() { 0 } else { xs[xs.len() - 1] };
        let l = self.lower_bound()?;
        let u = self.upper_bound()?;
        let j = iproject(i, l, u);
        Ok((j, (l, u)))
    }

    fn lower_bound(&self) -> Result<i32> {
        let objective_function = |xs: &[f64],
                                  _: Option<&mut [f64]>,
                                  p: &mut &DiscreteHomProblem<'a>|
         -> f64 {
            p.objective_function(&xs.to_vec().to_i()).unwrap()
        };

        let xs = self.past_opt(objective_function)?;
        Ok(xs[xs.len() - 1].ceil() as i32)
    }

    fn upper_bound(&self) -> Result<i32> {
        let objective_function = |xs: &[f64],
                                  _: Option<&mut [f64]>,
                                  p: &mut &DiscreteHomProblem<'a>|
         -> f64 {
            p.inverted_objective_function(&xs.to_vec().to_i()).unwrap()
        };

        let xs = self.past_opt(objective_function)?;
        Ok(xs[xs.len() - 1].ceil() as i32)
    }
}
