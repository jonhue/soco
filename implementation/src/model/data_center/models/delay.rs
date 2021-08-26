//! Delay model.

use noisy_float::prelude::*;

/// Average delay of a job processed on a server handling a total of
/// `number_of_jobs` jobs with average duration `mean_job_duration` using
/// the model of a M/GI/1 Processor Sharing Queue.
/// `delta` is the duration of a time slot.
/// Referred to as `d` in the paper.
pub fn average_delay(
    delta: f64,
    number_of_jobs: N64,
    mean_job_duration: N64,
) -> N64 {
    if number_of_jobs > n64(0.) && mean_job_duration > n64(0.) {
        let service_rate = n64(1.) / mean_job_duration;
        let arrival_rate = number_of_jobs / delta;
        if arrival_rate < service_rate {
            n64(1.) / (service_rate - arrival_rate)
        } else {
            n64(f64::INFINITY)
        }
    } else {
        n64(0.)
    }
}
