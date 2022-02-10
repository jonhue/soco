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

#[pyclass(name = "OptimalGraphSearchOptions")]
#[derive(Clone, Default)]
pub struct Options {
    pub cache: Option<Cache<Vertice>>,
}
#[pymethods]
impl Options {
    #[new]
    fn constructor() -> Self {
        Options::default()
    }
}

/// Graph-Based Optimal Algorithm
pub fn optimal_graph_search<C, D>(
    p: IntegralSimplifiedSmoothedConvexOptimization<'_, C, D>,
    Options { cache }: Options,
    offline_options: OfflineOptions,
) -> Result<CachedPath<Cache<Vertice>>>
where
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    let max_bound = p.bounds.iter().max().unwrap();
    let values = Values {
        values: (0..=*max_bound).collect(),
        bound_indices: p.bounds.iter().map(|&m| m as usize).collect(),
    };
    debug!("starting with `{}` values", max_bound);
    graph_search(p, values, cache, offline_options)
}
