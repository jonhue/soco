use crate::distance::{DistanceGeneratingFn, euclidean, negative_entropy, norm_squared};
use crate::model::{ModelOutputFailure, ModelOutputSuccess};
use crate::algorithms::online::multi_dimensional::online_balanced_descent::primal::{pobd, Options as PrimalOptions};
use crate::numerics::convex_optimization::find_minimizer_of_hitting_cost;
use crate::config::{Config};
use crate::algorithms::online::{FractionalStep, OnlineAlgorithm, Step};
use crate::problem::{FractionalSmoothedConvexOptimization, Online};
use crate::result::{Failure, Result};
use crate::schedule::FractionalSchedule;
use crate::utils::assert;
use pyo3::prelude::*;

#[pyclass]
#[derive(Clone)]
pub struct Options {
    /// Convexity parameter. Chosen such that $f_t(x) \geq f_t(v_t) + \frac{m}{2} \norm{x - v_t}_2^2$ where $v_t$ is the minimizer of $f_t$.
    pub m: f64,
    /// Controls the size of the step towards the minimizer. $\mu > 0$. Defaults to $1$.
    pub mu: f64,
    /// Balance parameter in OBD. $\gamma > 0$. Defaults to $1$.
    pub gamma: f64,
    /// Distance-generating function.
    pub h: DistanceGeneratingFn<f64>,
}
impl Default for Options {
    fn default() -> Self {
        unimplemented!()
    }
}
#[pymethods]
impl Options {
    #[staticmethod]
    pub fn euclidean_squared(m: f64, mu: f64, gamma: f64) -> Self {
        Options {
            m,
            mu,
            gamma,
            h: norm_squared(euclidean()),
        }
    }

    #[staticmethod]
    pub fn negative_entropy(m: f64, mu: f64, gamma: f64) -> Self {
        Options {
            m,
            mu,
            gamma,
            h: negative_entropy(),
        }
    }
}

/// Greedy Online Balanced Descent
pub fn gobd<C, D>(
    o: Online<FractionalSmoothedConvexOptimization<C, D>>,
    t: i32,
    xs: &FractionalSchedule,
    _: (),
    Options { m, mu, gamma, h }: Options,
) -> Result<FractionalStep<()>>
where
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    assert(o.w == 0, Failure::UnsupportedPredictionWindow(o.w))?;

    let v = Config::new(
        find_minimizer_of_hitting_cost(
            t,
            o.p.hitting_cost.clone(),
            o.p.bounds.clone(),
        )
        .0,
    );
    let Step(y, _) =
        pobd.next(o, xs, None, PrimalOptions { beta: gamma, h })?;

    let x = if mu * m.sqrt() >= 1. {
        v
    } else {
        mu * m.sqrt() * v + (1. - mu * m.sqrt()) * y
    };
    Ok(Step(x, None))
}
