use crate::online::Online;
use crate::online::OnlineSolution;
use crate::problem::DiscreteSmoothedConstantOptimization;
use crate::result::{Error, Result};
use crate::schedule::DiscreteSchedule;
use crate::schedule::Step;
use crate::utils::assert;

/// Memory.
pub type Memory = ();

/// Deterministic Online Algorithm
pub fn det<'a>(
    o: &'a Online<DiscreteSmoothedConstantOptimization>,
    xs: &DiscreteSchedule,
    ps: &Vec<Memory>,
) -> Result<OnlineSolution<Step<f64>, Memory>> {
    assert(o.w == 0, Error::UnsupportedPredictionWindow)?;
}
