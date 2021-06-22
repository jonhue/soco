use crate::algorithms::graph_search::Path;
use crate::algorithms::offline::multi_dimensional::graph_search::{
    graph_search, Values,
};
use crate::algorithms::offline::OfflineOptions;
use crate::problem::IntegralSimplifiedSmoothedConvexOptimization;
use crate::result::Result;

/// Graph-Based Optimal Algorithm
pub fn optimal_graph_search<'a>(
    p: &'a IntegralSimplifiedSmoothedConvexOptimization<'a>,
    offline_options: &OfflineOptions,
) -> Result<Path> {
    let all_values = build_all_values(p);
    graph_search(p, all_values, offline_options)
}

/// Computes all values.
fn build_all_values(
    p: &IntegralSimplifiedSmoothedConvexOptimization<'_>,
) -> Values {
    let mut all_values = vec![];
    for k in 0..p.d as usize {
        let mut values = vec![];
        for j in 0..=p.bounds[k] {
            values.push(j);
        }
        all_values.push(values);
    }
    all_values
}
