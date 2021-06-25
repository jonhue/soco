#![allow(clippy::float_cmp)]

use crate::algorithms::graph_search::Path;
use crate::algorithms::offline::multi_dimensional::{
    graph_search::graph_search, Values,
};
use crate::algorithms::offline::OfflineOptions;
use crate::problem::IntegralSimplifiedSmoothedConvexOptimization;
use crate::result::Result;

static DEFAULT_GAMMA: f64 = 1.1;

pub struct Options {
    /// `gamma > 1`. Default is `1.1`.
    pub gamma: Option<f64>,
}

/// Graph-Based Approximation Algorithm
pub fn approx_graph_search<'a>(
    p: &'a IntegralSimplifiedSmoothedConvexOptimization<'a>,
    options: &Options,
    offline_options: &OfflineOptions,
) -> Result<Path> {
    let gamma = options.gamma.unwrap_or(DEFAULT_GAMMA);
    let values = cache_bound_indices(build_values, &p.bounds, gamma);
    graph_search(p, values, offline_options)
}

/// Computes all values allowed by the approximation algorithm.
fn cache_bound_indices(
    build_values: impl Fn(i32, f64) -> Vec<i32>,
    bounds: &Vec<i32>,
    gamma: f64,
) -> Values {
    let max_bound = bounds.iter().max().unwrap();
    let values = build_values(*max_bound, gamma);
    let bound_indices = bounds
        .iter()
        .map(|m| values.iter().position(|j| j == m).unwrap())
        .collect();
    Values {
        values,
        bound_indices,
    }
}

fn build_values(bound: i32, gamma: f64) -> Vec<i32> {
    let mut vs: Vec<i32> = vec![0];

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
    }
    if !vs.contains(&bound) {
        vs.push(bound);
    }

    vs
}
