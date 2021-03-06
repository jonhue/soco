//! Offline Algorithms.

pub mod multi_dimensional;
pub mod uni_dimensional;

mod graph_search;

pub use graph_search::{Cache, CachedPath};

use crate::algorithms::Options;
use crate::model::{ModelOutputFailure, ModelOutputSuccess};
use crate::problem::Problem;
use crate::result::Result;
use crate::schedule::Schedule;
use pyo3::prelude::*;

/// Implementation of an offline algorithm.
///
/// * `R` - result
/// * `P` - problem
/// * `O` - options
///
/// Receives the arguments:
/// * `p` - problem instance
/// * `options` - algorithm options
/// * `offline_options` - general configuration of the offlien setting
pub trait OfflineAlgorithm<T, R, P, O, C, D>:
    Fn(P, O, OfflineOptions) -> Result<R>
where
    R: OfflineResult<T>,
    P: Problem<T, C, D>,
    O: Options<T, P, C, D>,
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    fn solve(
        &self,
        p: P,
        options: O,
        offline_options: OfflineOptions,
    ) -> Result<R> {
        self(p, options, offline_options)
    }

    fn solve_with_default_options(
        &self,
        p: P,
        offline_options: OfflineOptions,
    ) -> Result<R> {
        let options = O::default(&p);
        self.solve(p, options, offline_options)
    }
}
impl<T, R, P, O, C, D, F> OfflineAlgorithm<T, R, P, O, C, D> for F
where
    R: OfflineResult<T>,
    P: Problem<T, C, D>,
    O: Options<T, P, C, D>,
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
    F: Fn(P, O, OfflineOptions) -> Result<R>,
{
}

/// Configuration of offline algorithms.
#[pyclass]
#[derive(Clone, Debug)]
pub struct OfflineOptions {
    /// Compute inverted movement costs (SSCO only).
    #[pyo3(get, set)]
    pub inverted: bool,
    /// Compute the $\alpha$-unfair offline optimum.
    #[pyo3(get, set)]
    pub alpha: f64,
    /// Compute the $L$-constrained offline optimum (`convex_optimization` only).
    #[pyo3(get, set)]
    pub l: Option<f64>,
}
impl Default for OfflineOptions {
    fn default() -> Self {
        OfflineOptions {
            inverted: false,
            alpha: 1.,
            l: None,
        }
    }
}
impl OfflineOptions {
    pub fn inverted() -> Self {
        Self {
            inverted: true,
            ..Self::default()
        }
    }

    pub fn alpha_unfair(alpha: f64) -> Self {
        Self {
            alpha,
            ..Self::default()
        }
    }

    pub fn l_constrained(l: f64) -> Self {
        Self {
            l: Some(l),
            ..Self::default()
        }
    }
}
#[pymethods]
impl OfflineOptions {
    #[new]
    pub fn new(inverted: bool, alpha: f64, l: Option<f64>) -> Self {
        OfflineOptions { inverted, alpha, l }
    }
}

/// Result of an offline algorithm.
pub trait OfflineResult<T> {
    /// Resulting schedule.
    fn xs(self) -> Schedule<T>;
}

/// Result of an offline algorithm which only returns the obtained schedule.
pub struct PureOfflineResult<T> {
    /// Schedule.
    pub xs: Schedule<T>,
}
impl<T> OfflineResult<T> for PureOfflineResult<T> {
    fn xs(self) -> Schedule<T> {
        self.xs
    }
}
