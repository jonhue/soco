//! Multi-Dimensional Offline Algorithms.

pub mod approx_graph_search;
pub mod convex_optimization;
pub mod optimal_graph_search;
pub mod static_fractional;
pub mod static_integral;

mod graph_search;

pub use graph_search::Vertice;

/// Lists the number of possible values of a config, from smallest to largest
/// as well as the indices of the upper bound of each dimension.
#[derive(Clone, Debug)]
pub struct Values {
    values: Vec<i32>,
    bound_indices: Vec<usize>,
}
