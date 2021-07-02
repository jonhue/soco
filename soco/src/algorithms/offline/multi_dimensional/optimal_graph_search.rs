use crate::algorithms::graph_search::Path;
use crate::algorithms::offline::multi_dimensional::{
    graph_search::graph_search, Values,
};
use crate::problem::IntegralSimplifiedSmoothedConvexOptimization;
use crate::result::Result;

/// Graph-Based Optimal Algorithm
pub fn optimal_graph_search(
    p: IntegralSimplifiedSmoothedConvexOptimization<'_>,
    _: (),
    inverted: bool,
) -> Result<Path> {
    let max_bound = p.bounds.iter().max().unwrap();
    let values = Values {
        values: (0..=*max_bound).collect(),
        bound_indices: p.bounds.iter().map(|&m| m as usize).collect(),
    };
    graph_search(p, values, inverted)
}
