//! Helper functions for schedules.

/// Result of the Homogeneous Data-Center Right-Sizing problem.
/// Number of active servers from time 1 to time T.
pub type Schedule<T> = Vec<T>;
pub type DiscreteSchedule = Schedule<i32>;
pub type ContinuousSchedule = Schedule<f64>;

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
