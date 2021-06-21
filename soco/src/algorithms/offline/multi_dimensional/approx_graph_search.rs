use crate::algorithms::graph_search::Path;
use crate::algorithms::offline::multi_dimensional::duplicate_and_push_to_all;
use crate::algorithms::offline::multi_dimensional::graph_search::graph_search;
use crate::algorithms::offline::OfflineOptions;
use crate::config::{Config, IntegralConfig};
use crate::problem::IntegralSimplifiedSmoothedConvexOptimization;
use crate::result::Result;

static DEFAULT_GAMMA: f64 = 1.1;

pub struct Options {
    /// `gamma > 1`. Default is `2`.
    pub gamma: Option<f64>,
}

/// Graph-Based Approximation Algorithm
pub fn approx_graph_search<'a>(
    p: &'a IntegralSimplifiedSmoothedConvexOptimization<'a>,
    options: &Options,
    offline_options: &OfflineOptions,
) -> Result<Path> {
    let gamma = options.gamma.unwrap_or(DEFAULT_GAMMA);
    let configs = build_configs(p, gamma);
    graph_search(p, &configs, offline_options)
}

/// Computes all configurations examined by the approximation algorithm.
fn build_configs(
    p: &IntegralSimplifiedSmoothedConvexOptimization<'_>,
    gamma: f64,
) -> Vec<IntegralConfig> {
    let mut configs: Vec<IntegralConfig> = vec![Config::empty()];
    for k in 0..p.d {
        let vs = build_values(p.bounds[k as usize], gamma);

        let base = configs.clone();
        configs = vec![];
        for v in vs {
            duplicate_and_push_to_all(&mut configs, &base, v);
        }
    }
    configs
}

fn build_values(bound: i32, gamma: f64) -> Vec<i32> {
    if (bound as f64) < gamma.powi(bound) {
        build_values_via_exp(bound, gamma)
    } else {
        build_values_via_log(bound, gamma)
    }
}

fn build_values_via_exp(bound: i32, gamma: f64) -> Vec<i32> {
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
    }
    if !vs.contains(&bound) {
        vs.push(bound);
    }

    vs
}

fn build_values_via_log(bound: i32, gamma: f64) -> Vec<i32> {
    let mut vs: Vec<i32> = vec![0, 1];

    for j in 2..=bound {
        let l = (j as f64 - 1.).log(gamma);
        let u = (j as f64 + 1.).log(gamma);
        if l.floor() as i32 != u.floor() as i32
            || l.ceil() as i32 != u.ceil() as i32
        {
            vs.push(j);
        }
    }
    if !vs.contains(&bound) {
        vs.push(bound);
    }

    vs
}
