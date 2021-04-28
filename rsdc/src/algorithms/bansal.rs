use crate::problem::{
    ContinuousHomProblem, ContinuousSchedule, Online, OnlineSolution,
};

pub type Memory = ();

impl<'a> Online<ContinuousHomProblem<'a>> {
    /// Deterministic Online Algorithm
    pub fn bansal(
        &self,
        xs: &ContinuousSchedule,
        _: &Vec<Memory>,
    ) -> OnlineSolution<f64, Memory> {
    }
}