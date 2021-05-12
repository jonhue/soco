use std::collections::HashMap;

use crate::schedule::DiscreteSchedule;

/// The minimal cost from some initial vertice alongside the shortest path to the final vertice.
pub type Path = (DiscreteSchedule, f64);
/// Maps a vertice to its minimal cost from some initial vertice alongside the shortest path.
pub type Paths = HashMap<(i32, i32), Path>;
