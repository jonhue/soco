//! Non-continuous or non-smooth points of functions.

use noisy_float::prelude::*;
use pyo3::prelude::*;
use rayon::slice::ParallelSliceMut;
use std::sync::Arc;

/// Sorted non-continuous or non-smooth points of a function.
#[pyclass]
#[derive(Clone)]
pub struct Breakpoints {
    /// Finite vector of breakpoints.
    pub bs: Vec<N64>,
    /// Function which given a breakpoint returns the previous and next breakpoints (until there are none), respectively.
    /// The function is called to obtain the next breakpoint until the piecewise integrals converge to $0$ or the entire integral was integrated.
    #[allow(clippy::type_complexity)]
    pub next:
        Option<Arc<dyn Fn(f64) -> (Option<f64>, Option<f64>) + Send + Sync>>,
}
impl Breakpoints {
    /// Empty set of breakpoints.
    pub fn empty() -> Self {
        Breakpoints {
            bs: vec![],
            next: None,
        }
    }

    /// Generate breakpoints from a finite vector of breakpoints $bs$.
    pub fn from(bs: Vec<f64>) -> Self {
        Self::empty().add(&bs)
    }

    /// Breakpoints on a grid with a mesh width of $d$.
    pub fn grid(d: f64) -> Self {
        Breakpoints {
            bs: vec![],
            next: Some(Arc::new(move |b| {
                (Some(b.ceil() - d), Some(b.floor() + d))
            })),
        }
    }

    /// Adds breakpoints in $bs$ to the set of breakpoints, unless already included.
    /// Note: The caller must ensure that there are no duplicates within $bs$.
    pub fn add(&self, bs: &Vec<f64>) -> Self {
        let new_bs = bs
            .iter()
            .filter(|&&b| !self.bs.contains(&n64(b)))
            .map(|&b| n64(b));
        let mut breakpoints = self.clone();
        breakpoints.bs.extend(new_bs);
        breakpoints.bs.par_sort_unstable();
        breakpoints
    }
}
impl Default for Breakpoints {
    fn default() -> Self {
        Breakpoints::empty()
    }
}
#[pymethods]
impl Breakpoints {
    #[new]
    fn constructor(bs: Vec<f64>) -> Self {
        Breakpoints::from(bs)
    }
}
