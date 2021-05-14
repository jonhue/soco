use crate::algorithms::graph_search::Path;
use crate::algorithms::offline::multi_dimensional::graph_search::graph_search;
use crate::algorithms::offline::OfflineOptions;
use crate::config::Config;
use crate::problem::IntegralSmoothedConvexOptimization;
use crate::result::Result;
use crate::utils::duplicate_and_push_to_all;

/// Graph-Based Optimal Integral Algorithm
pub fn optimal_graph_search<'a>(
    p: &'a IntegralSmoothedConvexOptimization<'a>,
    offline_options: &OfflineOptions,
) -> Result<Path> {
    let configs = build_configs(p);
    graph_search(p, &configs, offline_options)
}

/// Computes all configurations.
fn build_configs(
    p: &IntegralSmoothedConvexOptimization<'_>,
) -> Vec<Config<i32>> {
    let mut configs: Vec<Vec<i32>> = vec![vec![]];
    for k in 0..p.d {
        let base = configs.clone();
        configs = vec![];

        for j in 0..=p.bounds[k as usize] {
            duplicate_and_push_to_all(&mut configs, &base, j);
        }
    }
    configs.into_iter().map(Config::new).collect()
}
