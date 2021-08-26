use super::next;
use crate::algorithms::online::{FractionalStep, Step};
use crate::config::{Config, FractionalConfig};
use crate::model::{ModelOutputFailure, ModelOutputSuccess};
use crate::problem::{
    DefaultGivenOnlineProblem, FractionalSimplifiedSmoothedConvexOptimization,
    Online,
};
use crate::result::Result;
use crate::schedule::FractionalSchedule;
use pyo3::prelude::*;
use serde_derive::{Deserialize, Serialize};

#[pyclass]
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Memory {
    /// Last initial configs for each iteration $k$.
    pub prev_x: Vec<FractionalConfig>,
}
impl<'a, C, D>
    DefaultGivenOnlineProblem<
        f64,
        FractionalSimplifiedSmoothedConvexOptimization<'a, C, D>,
        C,
        D,
    > for Memory
where
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    fn default(
        o: &Online<FractionalSimplifiedSmoothedConvexOptimization<C, D>>,
    ) -> Self {
        Self {
            prev_x: vec![Config::repeat(0., o.p.d); o.w as usize + 1],
        }
    }
}

/// Averaging Fixed Horizon Control
pub fn afhc<C, D>(
    o: Online<FractionalSimplifiedSmoothedConvexOptimization<C, D>>,
    t: i32,
    _: &FractionalSchedule,
    memory: Memory,
    _: (),
) -> Result<FractionalStep<Memory>>
where
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    let mut x = Config::repeat(0., o.p.d);
    let mut prev_x = vec![];
    for k in 1..=o.w + 1 {
        let (new_prev_x, new_x) =
            next(k, o.clone(), t, memory.prev_x[k as usize - 1].clone());
        prev_x.push(new_prev_x);
        x = x + new_x;
    }
    Ok(Step(x / (o.w + 1) as f64, Some(Memory { prev_x })))
}
