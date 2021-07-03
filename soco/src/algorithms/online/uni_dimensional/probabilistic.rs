use crate::algorithms::online::{FractionalStep, Step};
use crate::breakpoints::Breakpoints;
use crate::config::Config;
use crate::numerics::convex_optimization::{
    find_unbounded_minimizer_of_hitting_cost, maximize, minimize,
};
use crate::numerics::finite_differences::{derivative, second_derivative};
use crate::numerics::quadrature::piecewise::piecewise_integral;
use crate::problem::{
    DefaultGivenProblem, FractionalSimplifiedSmoothedConvexOptimization, Online,
};
use crate::result::{Failure, Result};
use crate::schedule::FractionalSchedule;
use crate::utils::assert;
use std::sync::Arc;

/// Probability distribution.
type Distribution<'a> = Arc<dyn Fn(f64) -> f64 + 'a>;

#[derive(Clone)]
pub struct Memory<'a> {
    /// Probability distribution.
    p: Distribution<'a>,
    /// List of non-continuous or non-smooth points of the probability distribution.
    breakpoints: Vec<f64>,
}
impl<'a> DefaultGivenProblem<FractionalSimplifiedSmoothedConvexOptimization<'a>>
    for Memory<'_>
{
    fn default(p: &FractionalSimplifiedSmoothedConvexOptimization<'a>) -> Self {
        let m = p.bounds[0];
        Memory {
            p: Arc::new(move |x| if 0. <= x && x <= m { 1. / m } else { 0. }),
            breakpoints: vec![0., m],
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

    let x = expected_value(&breakpoints, &prev_p, x_l, x_r)?;

    let p: Arc<dyn Fn(f64) -> f64> = Arc::new(move |x| {
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
        p,
        breakpoints: prev_m.breakpoints.clone(),
    };
    m.breakpoints.extend(&vec![x_l, x_r]);

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
    let bounds = vec![(x_m, o.p.bounds[0])];
    let objective = |x: &[f64]| x[0];
    let constraint = Arc::new(|x: &[f64]| -> f64 {
        let f = |x| o.p.hitting_cost.call_unbounded(t, Config::single(x));
        let g = derivative(f, x[0]) - derivative(f, x_m);
        let h =
            piecewise_integral(breakpoints, x[0], f64::INFINITY, |x| prev_p(x))
                .unwrap();
        g / 2. - o.p.switching_cost[0] * h
    });
    let init = vec![x_m];

    let (x, _) =
        maximize(objective, &bounds, Some(init), vec![], vec![constraint])?;
    Ok(x[0])
}

/// Determines `x_l` with a convex optimization.
fn find_left_bound(
    o: &Online<FractionalSimplifiedSmoothedConvexOptimization<'_>>,
    t: i32,
    breakpoints: &Breakpoints,
    prev_p: &Distribution,
    x_m: f64,
) -> Result<f64> {
    let bounds = vec![(0., x_m)];
    let objective = |x: &[f64]| x[0];
    let constraint = Arc::new(|x: &[f64]| -> f64 {
        let f = |x| o.p.hitting_cost.call_unbounded(t, Config::single(x));
        let g = derivative(f, x_m) - derivative(f, x[0]);
        let h = piecewise_integral(breakpoints, f64::NEG_INFINITY, x[0], |x| {
            prev_p(x)
        })
        .unwrap();
        o.p.switching_cost[0] * h - g / 2.
    });
    let init = vec![x_m];

    let (x, _) =
        minimize(objective, &bounds, Some(init), vec![], vec![constraint])?;
    Ok(x[0])
}

fn expected_value(
    breakpoints: &Breakpoints,
    prev_p: &Distribution,
    from: f64,
    to: f64,
) -> Result<f64> {
    piecewise_integral(breakpoints, from, to, |x| x * prev_p(x))
}
