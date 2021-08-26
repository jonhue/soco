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
    /// Revenue loss factor. `gamma >= 0`.
    #[pyo3(get, set)]
    gamma: f64,
    /// Minimal detectable delay of a job type. `delta >= 0`.
    #[pyo3(get, set)]
    delta: f64,
}
impl Default for MinimalDetectableDelayRevenueLossModel {
    fn default() -> Self {
        MinimalDetectableDelayRevenueLossModel {
            gamma: 1.,
            delta: 0.,
        }
    }
}
#[pymethods]
impl MinimalDetectableDelayRevenueLossModel {
    #[new]
    fn constructor(gamma: f64, delta: f64) -> Self {
        MinimalDetectableDelayRevenueLossModel { gamma, delta }
    }
}

impl RevenueLossModel {
    /// Revenue loss if jobs of some type have average delay `delay` during time slot `t`.
    /// Referred to as `r` in the paper.
    pub fn loss(&self, _t: i32, job_type: &JobType, delay: N64) -> N64 {
        match self {
            RevenueLossModel::MinimalDetectableDelay(models) => {
                let model = &models[&job_type.key];
                n64(model.gamma) * pos(delay - n64(model.delta))
            }
        }
    }
}
