use super::graph_search::Vertice;
use crate::algorithms::offline::graph_search::{Cache, CachedPath};
use crate::algorithms::offline::multi_dimensional::{
    graph_search::graph_search, Values,
};
use crate::algorithms::offline::OfflineOptions;
use crate::model::{ModelOutputFailure, ModelOutputSuccess};
use crate::problem::IntegralSimplifiedSmoothedConvexOptimization;
use crate::result::Result;
use log::debug;
use pyo3::prelude::*;
use rayon::slice::ParallelSliceMut;

#[pyclass(name = "ApproxGraphSearchOptions")]
#[derive(Clone)]
pub struct Options {
    pub cache: Option<Cache<Vertice>>,
    /// `gamma > 1`. Default is `1.1`.
    #[pyo3(get, set)]
    pub gamma: f64,
}
impl Default for Options {
    fn default() -> Self {
        Options {
            cache: None,
            gamma: 1.1,
        }
    }
}
impl Options {
    pub fn new(gamma: f64) -> Self {
        Options {
            gamma,
            ..Options::default()
        }
    }
}
#[pymethods]
impl Options {
    #[new]
    fn constructor(gamma: f64) -> Self {
        Options { cache: None, gamma }
    }
}

/// Graph-Based Polynomial-Time Approximation Scheme
pub fn approx_graph_search<C, D>(
    p: IntegralSimplifiedSmoothedConvexOptimization<'_, C, D>,
    Options { cache, gamma }: Options,
    offline_options: OfflineOptions,
) -> Result<CachedPath<Cache<Vertice>>>
where
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    let values = cache_bound_indices(build_values, &p.bounds, gamma);
    debug!("starting with `{}` values", values.values.len());
    graph_search(p, values, cache, offline_options)
}

/// Computes all values allowed by the approximation algorithm.
fn cache_bound_indices(
    build_values: impl Fn(&Vec<i32>, f64) -> Vec<i32>,
    bounds: &Vec<i32>,
    gamma: f64,
) -> Values {
    let values = build_values(bounds, gamma);
    let bound_indices = bounds
        .iter()
        .map(|m| values.iter().position(|j| j == m).unwrap())
        .collect();
    Values {
        values,
        bound_indices,
    }
}

fn build_values(bounds: &Vec<i32>, gamma: f64) -> Vec<i32> {
    let max_bound = *bounds.iter().max().unwrap();
    let mut vs: Vec<i32> = vec![0];

    let mut i = 1;
    loop {
        let l = gamma.powi(i).floor() as i32;
        if l > max_bound {
            break;
        }
        if !vs.contains(&l) {
            vs.push(l);
        }

        let u = gamma.powi(i).ceil() as i32;
        if u > max_bound {
            break;
        }
        if !vs.contains(&u) {
            vs.push(u);
        }

        i += 1;
    }
    for &bound in bounds {
        if !vs.contains(&bound) {
            vs.push(bound);
        }
    }

    vs.par_sort_unstable();
    vs
}
