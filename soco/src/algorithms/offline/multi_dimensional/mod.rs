//! Multi-Dimensional Offline Algorithms.

use crate::config::Config;

pub mod approx_graph_search;
pub mod optimal_graph_search;

mod graph_search;

/// Clones `base`, appends `j` to all configs within `base`, and extends `bag` with the updated `base`.
pub fn duplicate_and_push_to_all<T>(
    bag: &mut Vec<Config<T>>,
    base: &Vec<Config<T>>,
    j: T,
) where
    T: Copy,
{
    let mut tmp = base.clone();
    for x in tmp.iter_mut() {
        x.push(j);
    }
    bag.extend(tmp);
}
