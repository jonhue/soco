//! Revenue loss model.

use crate::model::data_center::model::JobType;
use crate::utils::pos;
use noisy_float::prelude::*;
use pyo3::prelude::*;
use std::collections::HashMap;

/// Revenue loss model. Parameters are provided separately for each job type.
#[derive(Clone, FromPyObject)]
pub enum RevenueLossModel {
    /// Linear loss based on average delay exceeding the minimal detectable delay.
    MinimalDetectableDelay(
        HashMap<String, MinimalDetectableDelayRevenueLossModel>,
    ),
}

#[pyclass]
#[derive(Clone)]
pub struct MinimalDetectableDelayRevenueLossModel {
    /// Minimal detectable delay of a job type.
    delta: f64,
}
impl Default for MinimalDetectableDelayRevenueLossModel {
    fn default() -> Self {
        MinimalDetectableDelayRevenueLossModel { delta: 0. }
    }
}

impl RevenueLossModel {
    /// Revenue loss if jobs of some type have average delay `delay` during time slot `t`.
    /// Referred to as `r` in the paper.
    pub fn loss(&self, _t: i32, job_type: &JobType, delay: R64) -> R64 {
        match self {
            RevenueLossModel::MinimalDetectableDelay(models) => {
                let model = &models[&job_type.key];
                pos(delay - model.delta)
            }
        }
    }
}
