use rand::{thread_rng, Rng};

use crate::algorithms::bansal::det::Memory as RandMemory;
use crate::problem::{
    ContinuousHomProblem, DiscreteSchedule, Online, OnlineSolution,
};
use crate::schedule::ExtendedSchedule;
use crate::utils::{fproject, frac};

/// Continuous number of servers as determined by `bansal`; memory of `bansal`.
type Memory<'a> = (f64, RandMemory<'a>);

impl<'a> Online<ContinuousHomProblem<'a>> {
    /// Discrete Randomized Online Algorithm
    ///
    /// Note: Relax discrete problem to continuous problem before use!
    pub fn irand(
        &'a self,
        xs: &DiscreteSchedule,
        ms: &Vec<Memory<'a>>,
    ) -> OnlineSolution<i32, Memory<'a>> {
        let det_ms = ms.iter().map(|m| m.1.clone()).collect();
        let (y, det_m) = self.det(&xs.to_f(), &det_ms);

        let prev_x = if xs.is_empty() { 0 } else { xs[xs.len() - 1] };
        let prev_y = if ms.is_empty() {
            0.
        } else {
            ms[ms.len() - 1].0
        };

        let x = next(prev_x, prev_y, y);

        (x, (y, det_m))
    }
}

fn next(prev_x: i32, prev_y: f64, y: f64) -> i32 {
    #[allow(clippy::collapsible_else_if)]
    // Number of active servers increases (or remains the same).
    if prev_y <= y {
        if prev_x == y.ceil() as i32 {
            prev_x
        } else {
            let prev_y_proj = fproject(prev_y, y.floor(), y.ceil());
            let p = (y - prev_y_proj) / (1. - frac(prev_y_proj));

            let r = sample_uniform();
            if r <= p {
                y.ceil() as i32
            } else {
                y.floor() as i32
            }
        }
    }
    // Number of active servers decreases.
    else {
        if prev_x == y.floor() as i32 {
            prev_x
        } else {
            let prev_y_proj = fproject(prev_y, y.floor(), y.ceil());
            let p = (prev_y_proj - y) / frac(prev_y_proj);

            let r = sample_uniform();
            if r <= p {
                y.floor() as i32
            } else {
                y.ceil() as i32
            }
        }
    }
}

fn sample_uniform() -> f64 {
    let mut rng = thread_rng();
    rng.gen_range(0.0..=1.0)
}
