use crate::algorithms::graph_search::Path;
use crate::algorithms::offline::multi_dimensional::graph_search::graph_search;
use crate::config::Config;
use crate::problem::IntegralSmoothedConvexOptimization;
use crate::result::Result;
use crate::utils::duplicate_and_push_to_all;

static GAMMA: f64 = (1. + f64::EPSILON) / 2.;

/// Graph-Based Polynomial-Time Integral Approximation Algorithm
pub fn approx_graph_search<'a>(
    p: &'a IntegralSmoothedConvexOptimization<'a>,
) -> Result<Path> {
    let configs = build_configs(p);
    graph_search(p, &configs)
}

/// Computes all configurations examined by the approximation algorithm.
fn build_configs(
    p: &IntegralSmoothedConvexOptimization<'_>,
) -> Vec<Config<i32>> {
    let mut configs: Vec<Vec<i32>> = vec![vec![]];
    for k in 1..=p.d {
        let bound = p.bounds[k as usize];
        let base = configs.clone();
        configs = vec![];

        duplicate_and_push_to_all(&mut configs, &base, 0);

        for i in 0..=bound {
            let l = GAMMA.powi(i).floor() as i32;
            if l > bound {
                break;
            }
            duplicate_and_push_to_all(&mut configs, &base, l);

            let u = GAMMA.powi(i).ceil() as i32;
            if u > bound {
                break;
            }
            duplicate_and_push_to_all(&mut configs, &base, u);
        }

        duplicate_and_push_to_all(&mut configs, &base, bound);
    }
    configs.into_iter().map(Config::new).collect()
}
