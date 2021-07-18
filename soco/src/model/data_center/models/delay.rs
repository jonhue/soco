//! Delay model.

/// Delay model.
#[derive(Clone)]
pub enum DelayModel {
    /// M/GI/1 Processor Sharing Queue.
    ProcessorSharingQueue,
}

impl DelayModel {
    /// Average delay of a sub job processed on a server handling a total of `l`
    /// sub jobs when the time slot length is `delta`.
    /// Referred to as `d` in the paper.
    pub fn average_delay(&self, delta: f64, l: f64) -> f64 {
        match self {
            DelayModel::ProcessorSharingQueue => 1. / (delta - l),
        }
    }
}
