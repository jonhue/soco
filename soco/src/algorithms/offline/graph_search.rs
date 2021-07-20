use super::OfflineResult;
use crate::schedule::IntegralSchedule;
use std::collections::HashMap;

/// The minimal cost from some initial vertice alongside the shortest path to the final vertice.
#[derive(Clone, Debug, PartialEq)]
pub struct Path {
    pub xs: IntegralSchedule,
    pub cost: f64,
}
impl OfflineResult<i32> for Path {
    fn xs(self) -> IntegralSchedule {
        self.xs
    }
}

/// Maps a vertice to its minimal cost from some initial vertice alongside the shortest path.
pub type Paths<T> = HashMap<T, Path>;
