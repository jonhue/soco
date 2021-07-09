use crate::cost::data_center::model::JobType;
use crate::utils::pos;
use std::collections::HashMap;

/// Revenue loss model.
pub enum RevenueLossModel {
    /// Linear loss based on average delay exceeding the minimal detectable delay.
    MinimalDetectableDelay(HashMap<String, MinimalDetectableDelay>),
}

pub struct MinimalDetectableDelay {
    /// Minimal detectable delay of a job type.
    delta: f64,
}

impl RevenueLossModel {
    /// Revenue loss if jobs of type `i` have average delay `d` during time slot `t`.
    /// Referred to as `r` in the paper.
    pub fn loss(&self, _t: i32, job_type: &JobType, delay: f64) -> f64 {
        match self {
            RevenueLossModel::MinimalDetectableDelay(models) => {
                let model = &models[&job_type.key];
                pos(delay - model.delta)
            }
        }
    }
}
