use crate::algorithms::online::{FractionalStep, Step};
use crate::breakpoints::Breakpoints;
use crate::config::Config;
use crate::model::{ModelOutputFailure, ModelOutputSuccess};
use crate::numerics::convex_optimization::find_minimizer_of_hitting_cost;
use crate::numerics::finite_differences::{derivative, second_derivative};
use crate::numerics::quadrature::piecewise::piecewise_integral;
use crate::numerics::roots::find_root;
use crate::problem::{FractionalSimplifiedSmoothedConvexOptimization, Online};
use crate::result::{Failure, Result};
use crate::schedule::FractionalSchedule;
use crate::utils::assert;
use log::debug;
use pyo3::prelude::*;
use serde_derive::{Deserialize, Serialize};
use std::sync::Arc;

const EPSILON: f64 = 1e-5;

/// Probability distribution.
type Distribution<'a> = Arc<dyn Fn(f64) -> f64 + Send + Sync + 'a>;

#[derive(Clone, Derivative, Deserialize, Serialize)]
#[derivative(Debug)]
pub struct Memory<'a> {
    /// Probability distribution.
    #[serde(skip, default = "default_p")]
    #[derivative(Debug = "ignore")]
    pub p: Distribution<'a>,
    /// List of non-continuous or non-smooth points of the probability distribution.
    pub breakpoints: Vec<f64>,
}
fn default_p<'a>() -> Distribution<'a> {
    Arc::new(|_| {
        panic!("This is a dummy distribution returned after deserializing the memory struct.");
    })
}
impl Default for Memory<'_> {
    fn default() -> Self {
        Memory {
            p: Arc::new(|x| {
                #[allow(clippy::manual_range_contains)]
                if 0. <= x && x <= EPSILON {
                    1. / EPSILON
                } else {
                    0.
                }
            }),
            breakpoints: vec![0., EPSILON],
        }
    }
}
impl IntoPy<PyObject> for Memory<'_> {
    fn into_py(self, py: Python) -> PyObject {
        self.breakpoints.into_py(py)
    }
}

#[pyclass]
#[derive(Clone)]
pub struct Options {
    /// Breakpoints of piecewise linear hitting costs.
    #[pyo3(get, set)]
    pub breakpoints: Breakpoints,
}
impl Default for Options {
    fn default() -> Self {
        Options {
            breakpoints: Breakpoints::empty(),
        }
    }
}
#[pymethods]
impl Options {
    #[new]
    fn constructor(breakpoints: Breakpoints) -> Self {
        Options { breakpoints }
    }
}

/// Probabilistic Algorithm
///
/// Assumes that the hitting costs are either smooth, i.e. infinitely many times continuously differentiable,
/// or piecewise linear in which case the breakpoints must be provided through the options.
pub fn probabilistic<'a, C, D>(
    o: Online<FractionalSimplifiedSmoothedConvexOptimization<'a, C, D>>,
    t: i32,
    _: &FractionalSchedule,
    prev_m: Memory<'a>,
    options: Options,
) -> Result<FractionalStep<Memory<'a>>>
where
    C: ModelOutputSuccess + 'a,
    D: ModelOutputFailure + 'a,
{
    assert(o.w == 0, Failure::UnsupportedPredictionWindow(o.w))?;
    assert(o.p.d == 1, Failure::UnsupportedProblemDimension(o.p.d))?;

    let breakpoints = options.breakpoints.add(&prev_m.breakpoints);
    let prev_p = prev_m.p;

    let x_m = find_minimizer_of_hitting_cost(
        t,
        o.p.hitting_cost.clone(),
        vec![(0., o.p.bounds[0])],
    )
    .0[0];
    debug!("determined minimizer {}", x_m);

    let x_r = find_right_bound(&o, t, &breakpoints, &prev_p, x_m);
    let x_l = find_left_bound(&o, t, &breakpoints, &prev_p, x_m);
    debug!("determined bounds {} and {}", x_l, x_r);

    let p: Distribution = Arc::new(move |x| {
        if x_l <= x && x <= x_r {
            prev_p(x)
                + second_derivative(
                    |x: f64| {
                        // needs to be unbounded for numerical approximations
                        o.p.hitting_cost
                            .call_certain(t, Config::single(x))
                            .cost
                            .raw()
                    },
                    x,
                )
                .raw()
                    / (2. * o.p.switching_cost[0])
        } else {
            0.
        }
    });
    let mut m = Memory {
        p: p.clone(),
        breakpoints: prev_m.breakpoints.clone(),
    };
    for b in [x_l, x_r] {
        if !m.breakpoints.contains(&b) {
            m.breakpoints.push(b)
        }
    }

    let mut x = expected_value(&breakpoints, &p, x_l, x_r);
    // if the expected value is below the lower bound or above the upper bound
    // (due to a numeric inaccuracy) we fallback to the minimizer
    if x < x_l || x > x_r {
        x = x_m;
    }
    debug!("determined next config: {:?}", x);

    Ok(Step(Config::single(x), Some(m)))
}

/// Determines $x_r$ with a convex optimization.
fn find_right_bound<C, D>(
    o: &Online<FractionalSimplifiedSmoothedConvexOptimization<'_, C, D>>,
    t: i32,
    breakpoints: &Breakpoints,
    prev_p: &Distribution,
    x_m: f64,
) -> f64
where
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    find_root((x_m, o.p.bounds[0]), |x| {
        let f = |x| {
            // needs to be unbounded for numerical approximations
            o.p.hitting_cost
                .call_certain(t, Config::single(x))
                .cost
                .raw()
        };
        derivative(f, x).raw()
            - 2. * o.p.switching_cost[0]
                * piecewise_integral(
                    breakpoints,
                    x,
                    f64::INFINITY,
                    prev_p.as_ref(),
                )
                .raw()
    })
    .raw()
}

/// Determines $x_l$ with a convex optimization.
fn find_left_bound<C, D>(
    o: &Online<FractionalSimplifiedSmoothedConvexOptimization<'_, C, D>>,
    t: i32,
    breakpoints: &Breakpoints,
    prev_p: &Distribution,
    x_m: f64,
) -> f64
where
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    find_root((0., x_m), |x| {
        let f = |x| {
            // needs to be unbounded for numerical approximations
            o.p.hitting_cost
                .call_certain(t, Config::single(x))
                .cost
                .raw()
        };
        2. * o.p.switching_cost[0]
            * piecewise_integral(
                breakpoints,
                f64::NEG_INFINITY,
                x,
                prev_p.as_ref(),
            )
            .raw()
            - derivative(f, x).raw()
    })
    .raw()
}

fn expected_value(
    breakpoints: &Breakpoints,
    prev_p: &Distribution,
    from: f64,
    to: f64,
) -> f64 {
    piecewise_integral(breakpoints, from, to, |x| x * prev_p(x)).raw()
}
