use std::collections::HashMap;

use crate::schedule::IntegralSchedule;

/// The minimal cost from some initial vertice alongside the shortest path to the final vertice.
#[derive(Clone, Debug, PartialEq)]
pub struct Path(pub IntegralSchedule, pub f64);

/// Maps a vertice to its minimal cost from some initial vertice alongside the shortest path.
pub type Paths<T> = HashMap<T, Path>;
