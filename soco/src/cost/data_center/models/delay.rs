/// Delay model.
pub enum DelayModel {
    /// M/GI/1 Processor Sharing Queue.
    ProcessorSharingQueue {
        /// Time slot length.
        delta: f64,
    },
}

impl DelayModel {
    /// Average delay of a job of type `i` processed on a server of type `k`
    /// during time slot `t` where the total load on the server is `l`.
    /// Referred to as `d` in the paper.
    pub fn average_delay(&self, l: f64) -> f64 {
        match self {
            DelayModel::ProcessorSharingQueue { delta } => 1. / (delta - l),
        }
    }
}
