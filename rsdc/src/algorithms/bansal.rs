//! Algorithms by Bansal et al.

use crate::problem::{
    ContinuousHomProblem, ContinuousSchedule, Online, OnlineSolution,
};

pub type Memory = ();

impl<'a> Online<ContinuousHomProblem<'a>> {
    /// Deterministic Online Algorithm
    pub fn bansal(
        &self,
        _: &ContinuousSchedule,
        _: &Vec<Memory>,
    ) -> OnlineSolution<f64, Memory> {
        (0., ())
    }
}
