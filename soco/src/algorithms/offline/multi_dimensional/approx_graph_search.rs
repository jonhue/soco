use crate::algorithms::graph_search::Path;
use crate::algorithms::offline::multi_dimensional::graph_search::graph_search;
use crate::config::Config;
use crate::problem::IntegralSmoothedConvexOptimization;
use crate::result::{Error, Result};
use crate::utils::duplicate_and_push_to_all;

static DEFAULT_GAMMA: f64 = 1.1;
static MAX_ITERATIONS: i32 = 1_000_000;

pub struct Options {
    /// `gamma > 1`. If `gamma` is too close to `1` the algorithm will not terminate. Default is `2`.
    pub gamma: Option<f64>,
}

/// Graph-Based Polynomial-Time Integral Approximation Algorithm
pub fn approx_graph_search<'a>(
    p: &'a IntegralSmoothedConvexOptimization<'a>,
    options: &Options,
) -> Result<Path> {
    let configs = build_configs(p, options.gamma.unwrap_or(DEFAULT_GAMMA))?;
    graph_search(p, &configs)
}

/// Computes all configurations examined by the approximation algorithm.
fn build_configs(
    p: &IntegralSmoothedConvexOptimization<'_>,
    gamma: f64,
) -> Result<Vec<Config<i32>>> {
    let mut configs: Vec<Vec<i32>> = vec![vec![]];
    for k in 0..p.d {
        let bound = p.bounds[k as usize];

        let mut vs: Vec<i32> = vec![0, 1];
        let mut i = 1;
        loop {
            let l = gamma.powi(i).floor() as i32;
            if l > bound {
                break;
            }
            if !vs.contains(&l) {
                vs.push(l);
            }

            let u = gamma.powi(i).ceil() as i32;
            if u > bound {
                break;
            }
            if !vs.contains(&u) {
                vs.push(u);
            }

            i += 1;
            if i > MAX_ITERATIONS {
                return Err(Error::GammaTooSmall);
            }
        }
        if !vs.contains(&bound) {
            vs.push(bound);
        }

        let base = configs.clone();
        configs = vec![];
        for v in vs {
            duplicate_and_push_to_all(&mut configs, &base, v);
        }
    }
    Ok(configs.into_iter().map(Config::new).collect())
}
