//! Delay model.

use noisy_float::prelude::*;
use pyo3::prelude::*;

/// Delay model.
#[derive(Clone, FromPyObject)]
pub enum DelayModel {
    /// M/GI/1 Processor Sharing Queue.
    ProcessorSharingQueue(ProcessorSharingQueueDelayModel),
}

#[pyclass]
#[derive(Clone)]
pub struct ProcessorSharingQueueDelayModel {
    /// Service rate. Should be set to the length of a time slot when using dynamic job duration.
    pub mu: f64,
}

impl DelayModel {
    /// Average delay of a sub job processed on a server handling a total of `l` sub jobs.
    /// Referred to as `d` in the paper.
    pub fn average_delay(&self, l: R64) -> R64 {
        match self {
            DelayModel::ProcessorSharingQueue(model) => {
                r64(1.) / (r64(model.mu) - l)
            }
        }
    }
}
