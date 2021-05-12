use crate::algorithms::graph_search::Path;
use crate::problem::DiscreteSmoothedConvexOptimization;
use crate::result::Result;

/// Graph-Based Optimal Discrete Algorithm
pub fn optimal_graph_search<'a>(
    _p: &'a DiscreteSmoothedConvexOptimization<'a>,
) -> Result<Path> {
    Ok((vec![], 0.))
}
