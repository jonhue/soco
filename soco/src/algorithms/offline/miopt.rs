use crate::problem::DiscreteSmoothedConvexOptimization;
use crate::result::Result;
use crate::schedule::DiscreteSchedule;

/// The minimal cost from some initial vertice alongside the shortest path to the final vertice.
pub type Path = (DiscreteSchedule, f64);

/// Multi-Dimensional Optimal Discrete Deterministic Offline Algorithm
pub fn miopt<'a>(
    _p: &'a DiscreteSmoothedConvexOptimization<'a>,
) -> Result<Path> {
    Ok((vec![], 0.))
}
