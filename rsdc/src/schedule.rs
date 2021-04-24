//! Helper functions for schedules.

use crate::problem::{ContinuousSchedule, DiscreteSchedule};

pub trait DiscretizableSchedule {
    fn to_i(&self) -> DiscreteSchedule;
}

impl DiscretizableSchedule for ContinuousSchedule {
    fn to_i(&self) -> DiscreteSchedule {
        self.iter().map(|&x| x.ceil() as i32).collect()
    }
}
