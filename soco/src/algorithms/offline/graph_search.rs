use super::OfflineResult;
use crate::schedule::IntegralSchedule;
use serde_derive::{Deserialize, Serialize};
use std::{collections::HashMap, hash::Hash};

/// Resulting path alongside cache which can be used for subsequent iterations.
#[derive(Clone, Debug)]
pub struct CachedPath<C> {
    pub path: Path,
    pub cache: C,
}
impl<C> OfflineResult<i32> for CachedPath<C> {
    fn xs(self) -> IntegralSchedule {
        self.path.xs
    }
}

/// The minimal cost from some initial vertice alongside the shortest path to the final vertice.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Path {
    pub xs: IntegralSchedule,
    pub cost: f64,
}
impl OfflineResult<i32> for Path {
    fn xs(self) -> IntegralSchedule {
        self.xs
    }
}

/// Data structure to cache results of the algorithm up to some time slot `t`.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Cache<T>
where
    T: Eq + Hash,
{
    /// Time slot.
    pub t: i32,
    /// Computed paths up to time slot `t`.
    pub paths: Paths<T>,
}

/// Returns next initial time slot `t_init` and `paths` from cache.
pub fn read_cache<T>(
    cache: Option<Cache<T>>,
    default: impl Fn() -> (i32, Paths<T>),
) -> (i32, Paths<T>)
where
    T: Eq + Hash,
{
    match cache {
        Some(Cache { t: prev_t, paths }) => (prev_t + 1, paths),
        None => default(),
    }
}

/// Maps a vertice to its minimal cost from some initial vertice alongside the shortest path.
pub type Paths<T> = HashMap<T, Path>;
