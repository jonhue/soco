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

pub trait ExtendedSchedule {
    fn to_f(&self) -> ContinuousSchedule;
}

impl ExtendedSchedule for DiscreteSchedule {
    fn to_f(&self) -> ContinuousSchedule {
        self.iter().map(|&x| x as f64).collect()
    }
}
