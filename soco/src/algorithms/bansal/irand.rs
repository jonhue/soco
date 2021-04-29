use rand::{thread_rng, Rng};

use crate::algorithms::bansal::rand::Memory as RandMemory;
use crate::problem::{
    DiscreteHomProblem, DiscreteSchedule, Online, OnlineSolution,
};
use crate::schedule::ExtendedSchedule;
use crate::utils::{fproject, frac};

/// Continuous number of servers as determined by `bansal`; memory of `bansal`.
type Memory<'a> = (f64, RandMemory<'a>);

impl<'a> Online<DiscreteHomProblem<'a>> {
    /// Discrete Randomized Online Algorithm
    pub fn irand(
        &'a self,
        xs: &DiscreteSchedule,
        ms: &Vec<Memory<'a>>,
    ) -> OnlineSolution<i32, Memory<'a>> {
        let rand_ms = ms.iter().map(|m| m.clone().1).collect();
        let (y, rand_m) = self.to_f().rand(&xs.to_f(), &rand_ms);

        let prev_x = if xs.is_empty() { 0 } else { xs[xs.len() - 1] };
        let prev_y = if ms.is_empty() {
            0.
        } else {
            ms[ms.len() - 1].0
        };

        #[allow(clippy::collapsible_else_if)]
        // Number of active servers increases (or remains the same).
        let x = if prev_y <= y {
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
        };

        (x, (y, rand_m))
    }
}

fn sample_uniform() -> f64 {
    let mut rng = thread_rng();
    rng.gen_range(0.0..=1.0)
}
