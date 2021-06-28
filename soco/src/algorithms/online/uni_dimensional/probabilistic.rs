use crate::config::Config;
use crate::convex_optimization::{
    find_unbounded_minimizer_of_hitting_cost, maximize, minimize,
};
use crate::online::{FractionalStep, Online, Step};
use crate::problem::FractionalSimplifiedSmoothedConvexOptimization;
use crate::quadrature::integral;
use crate::result::{Error, Result};
use crate::schedule::FractionalSchedule;
use crate::utils::{assert, project};
use crate::TOLERANCE;
use bacon_sci::differentiate::{derivative, second_derivative};
use ordered_float::OrderedFloat;
use std::sync::Arc;

/// Probability distribution.
type Distribution<'a> = Arc<dyn Fn(f64) -> f64 + 'a>;

/// Sorted discontinuous points of the probability distribution.
type Breakpoints = Vec<OrderedFloat<f64>>;

/// Memory comprised of probability distributions and breakpoints.
#[derive(Clone)]
pub struct Memory<'a> {
    p: Distribution<'a>,
    breakpoints: Breakpoints,
}
impl Default for Memory<'_> {
    fn default() -> Self {
        Memory {
            p: Arc::new(|x| if (0. ..1.).contains(&x) { 1. } else { 0. }),
            breakpoints: vec![OrderedFloat(0.), OrderedFloat(1.)],
        }
    }
}

/// Probabilistic Algorithm
///
/// Assumes that the hitting costs are twice differentiable and that the
/// minimizer is unique and bounded.
pub fn probabilistic<'a>(
    o: Online<FractionalSimplifiedSmoothedConvexOptimization<'a>>,
    t: i32,
    _: &FractionalSchedule,
    prev_m: Memory<'a>,
    _: &(),
) -> Result<FractionalStep<Memory<'a>>> {
    assert(o.w == 0, Error::UnsupportedPredictionWindow)?;
    assert(o.p.d == 1, Error::UnsupportedProblemDimension)?;

    let upper_bound = o.p.bounds[0];

    let x_m =
        find_unbounded_minimizer_of_hitting_cost(o.p.d, t, &o.p.hitting_cost)?
            .0[0];
    assert(
        x_m != f64::INFINITY && x_m != f64::NEG_INFINITY,
        Error::MinimizerShouldBeBounded,
    )?;
    let x_r = find_right_bound(&o, t, &prev_m, x_m)?;
    let x_l = find_left_bound(&o, t, &prev_m, x_m)?;

    let prev_p = prev_m.p.clone();
    let p: Arc<dyn Fn(f64) -> f64> = Arc::new(move |x| {
        if x >= x_l && x <= x_r {
            prev_p(x)
                + second_derivative(
                    |x: f64| (o.p.hitting_cost)(t, Config::single(x)).unwrap(),
                    x,
                    TOLERANCE.powf(-0.25),
                ) / 2.
        } else {
            0.
        }
    });
    let breakpoints = add_breakpoints(&prev_m.breakpoints.clone(), x_l, x_r);
    let m = Memory { p, breakpoints };

    let x = project(expected_value(&prev_m, x_l, x_r)?, 0., upper_bound);
    Ok(Step(Config::single(x), Some(m)))
}

/// Determines `x_r` with a convex optimization.
fn find_right_bound(
    o: &Online<FractionalSimplifiedSmoothedConvexOptimization<'_>>,
    t: i32,
    prev_m: &Memory<'_>,
    x_m: f64,
) -> Result<f64> {
    let bounds = vec![(x_m, f64::INFINITY)];
    let objective = |x: &[f64]| x[0];
    let constraint = Arc::new(|x: &[f64]| -> f64 {
        let f = |x| (o.p.hitting_cost)(t, Config::single(x)).unwrap();
        let g = derivative(f, x[0], TOLERANCE) - derivative(f, x_m, TOLERANCE);
        let h = integrate(&prev_m.breakpoints, x[0], f64::INFINITY, |x| {
            (prev_m.p)(x)
        })
        .unwrap();
        g / 2. - h
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
    prev_m: &Memory<'_>,
    x_m: f64,
) -> Result<f64> {
    let bounds = vec![(f64::NEG_INFINITY, x_m)];
    let objective = |x: &[f64]| x[0];
    let constraint = Arc::new(|x: &[f64]| -> f64 {
        let f = |x| (o.p.hitting_cost)(t, Config::single(x)).unwrap();
        let g = derivative(f, x_m, TOLERANCE) - derivative(f, x[0], TOLERANCE);
        let h = integrate(&prev_m.breakpoints, f64::NEG_INFINITY, x[0], |x| {
            (prev_m.p)(x)
        })
        .unwrap();
        h - g / 2.
    });
    let init = vec![x_m];

    let (x, _) =
        minimize(objective, &bounds, Some(init), vec![], vec![constraint])?;
    Ok(x[0])
}

fn expected_value(m: &Memory, from: f64, to: f64) -> Result<f64> {
    integrate(&m.breakpoints, from, to, |x| x * (m.p)(x))
}

fn integrate(
    breakpoints: &Breakpoints,
    from: f64,
    to: f64,
    f_: impl Fn(f64) -> f64,
) -> Result<f64> {
    let f = |x| f_(x);

    let l = breakpoints[0].into_inner();
    let r = breakpoints[breakpoints.len() - 1].into_inner();

    let mut result = if l > from { integral(from, l, f)? } else { 0. };
    for i in 1..breakpoints.len() {
        let prev_b = breakpoints[i - 1].into_inner();
        let b = breakpoints[i].into_inner();
        result += integral(prev_b, b, f)?;
    }
    Ok(result + if r < to { integral(r, to, f)? } else { 0. })
}

fn add_breakpoints(
    breakpoints_: &Breakpoints,
    x_l: f64,
    x_r: f64,
) -> Breakpoints {
    let mut breakpoints = breakpoints_.clone();
    if !breakpoints.contains(&OrderedFloat(x_l)) {
        breakpoints.push(OrderedFloat(x_l));
    }
    if !breakpoints.contains(&OrderedFloat(x_r)) {
        breakpoints.push(OrderedFloat(x_r));
    }
    breakpoints.sort_unstable();
    breakpoints
}
