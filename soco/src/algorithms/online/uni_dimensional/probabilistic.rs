use crate::algorithms::online::{FractionalStep, Step};
use crate::breakpoints::Breakpoints;
use crate::config::Config;
use crate::numerics::bisection::bisection;
use crate::numerics::convex_optimization::find_unbounded_minimizer_of_hitting_cost;
use crate::numerics::finite_differences::{derivative, second_derivative};
use crate::numerics::quadrature::piecewise::piecewise_integral;
use crate::numerics::PRECISION;
use crate::problem::{FractionalSimplifiedSmoothedConvexOptimization, Online};
use crate::result::{Failure, Result};
use crate::schedule::FractionalSchedule;
use crate::utils::assert;
use std::sync::Arc;

const EPSILON: f64 = 1e-5;

/// Probability distribution.
type Distribution<'a> = Arc<dyn Fn(f64) -> f64 + 'a>;

#[derive(Clone)]
pub struct Memory<'a> {
    /// Probability distribution.
    p: Distribution<'a>,
    /// List of non-continuous or non-smooth points of the probability distribution.
    breakpoints: Vec<f64>,
}
impl<'a> Default for Memory<'_> {
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

#[derive(Clone)]
pub struct Options {
    /// Breakpoints of piecewise linear hitting costs.
    pub breakpoints: Breakpoints,
}
impl Default for Options {
    fn default() -> Self {
        Options {
            breakpoints: Breakpoints::empty(),
        }
    }
}

/// Probabilistic Algorithm
///
/// Assumes that the hitting costs are either smooth, i.e. infinitely many times continuously differentiable,
/// or piecewise linear in which case the breakpoints must be provided through the options.
pub fn probabilistic<'a>(
    o: Online<FractionalSimplifiedSmoothedConvexOptimization<'a>>,
    t: i32,
    _: &FractionalSchedule,
    prev_m: Memory<'a>,
    options: Options,
) -> Result<FractionalStep<Memory<'a>>> {
    assert(o.w == 0, Failure::UnsupportedPredictionWindow(o.w))?;
    assert(o.p.d == 1, Failure::UnsupportedProblemDimension(o.p.d))?;

    let breakpoints = options.breakpoints.add(&prev_m.breakpoints);
    let prev_p = prev_m.p;

    let x_m =
        find_unbounded_minimizer_of_hitting_cost(o.p.d, t, &o.p.hitting_cost)?
            .0[0];
    let x_r = find_right_bound(&o, t, &breakpoints, &prev_p, x_m)?;
    let x_l = find_left_bound(&o, t, &breakpoints, &prev_p, x_m)?;

    let p: Distribution = Arc::new(move |x| {
        if x >= x_l && x <= x_r {
            prev_p(x)
                + second_derivative(
                    |x: f64| {
                        o.p.hitting_cost.call_unbounded(t, Config::single(x))
                    },
                    x,
                ) / (2. * o.p.switching_cost[0])
        } else {
            0.
        }
    });
    let mut m = Memory {
        p: p.clone(),
        breakpoints: prev_m.breakpoints.clone(),
    };
    m.breakpoints.extend(&vec![x_l, x_r]);

    let x = expected_value(&breakpoints, &p, x_l, x_r)?;
    Ok(Step(Config::single(x), Some(m)))
}

/// Determines `x_r` with a convex optimization.
fn find_right_bound(
    o: &Online<FractionalSimplifiedSmoothedConvexOptimization<'_>>,
    t: i32,
    breakpoints: &Breakpoints,
    prev_p: &Distribution,
    x_m: f64,
) -> Result<f64> {
    if (x_m - o.p.bounds[0]).abs() < PRECISION {
        Ok(0.)
    } else {
        bisection((x_m, o.p.bounds[0]), |x| {
            let f = |x| o.p.hitting_cost.call_unbounded(t, Config::single(x));
            derivative(f, x) / 2.
                - o.p.switching_cost[0]
                    * piecewise_integral(breakpoints, x, f64::INFINITY, |x| {
                        prev_p(x)
                    })
                    .unwrap()
        })
    }
}

/// Determines `x_l` with a convex optimization.
fn find_left_bound(
    o: &Online<FractionalSimplifiedSmoothedConvexOptimization<'_>>,
    t: i32,
    breakpoints: &Breakpoints,
    prev_p: &Distribution,
    x_m: f64,
) -> Result<f64> {
    if (x_m - 0.).abs() < PRECISION {
        Ok(0.)
    } else {
        bisection((0., x_m), |x| {
            let f = |x| o.p.hitting_cost.call_unbounded(t, Config::single(x));
            o.p.switching_cost[0]
                * piecewise_integral(breakpoints, f64::NEG_INFINITY, x, |x| {
                    prev_p(x)
                })
                .unwrap()
                - derivative(f, x) / 2.
        })
    }
}

fn expected_value(
    breakpoints: &Breakpoints,
    prev_p: &Distribution,
    from: f64,
    to: f64,
) -> Result<f64> {
    piecewise_integral(breakpoints, from, to, |x| x * prev_p(x))
}
