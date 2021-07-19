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
    /// Capacity of a server, i.e. load a server can handle during a time slot.
    /// Should be set to the length of a time slot when using dynamic job duration.
    pub c: f64,
}

impl DelayModel {
    /// Average delay of a sub job processed on a server handling a total of `l` sub jobs.
    /// Referred to as `d` in the paper.
    pub fn average_delay(&self, l: N64) -> N64 {
        match self {
            DelayModel::ProcessorSharingQueue(model) => {
                // assert!(l <= n64(model.c));
                // if l == n64(0.) {
                //     n64(0.)
                // } else {
                //     n64(1.) / (n64(model.c) / l - l)
                // }
                n64(0.)
            }
        }
    }
}
