use crate::algorithms::offline::graph_search::Path;
use crate::algorithms::offline::multi_dimensional::{
    graph_search::graph_search, Values,
};
use crate::problem::IntegralSimplifiedSmoothedConvexOptimization;
use crate::result::Result;
use pyo3::prelude::*;

#[pyclass]
#[derive(Clone)]
pub struct Options {
    /// `gamma > 1`. Default is `1.1`.
    #[pyo3(get, set)]
    pub gamma: f64,
}
impl Default for Options {
    fn default() -> Self {
        Options { gamma: 1.1 }
    }
}
#[pymethods]
impl Options {
    #[new]
    fn constructor(gamma: f64) -> Self {
        Options { gamma }
    }
}

/// Graph-Based Polynomial-Time Approximation Scheme
pub fn approx_graph_search(
    p: IntegralSimplifiedSmoothedConvexOptimization<'_>,
    options: Options,
    inverted: bool,
) -> Result<Path> {
    let values = cache_bound_indices(build_values, &p.bounds, options.gamma);
    graph_search(p, values, inverted)
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

    vs.sort_unstable();
    vs
}
