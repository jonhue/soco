use crate::config::Config;
use crate::convex_optimization::find_minimizer_of_hitting_cost;
use crate::convex_optimization::{maximize, minimize};
use crate::online::{FractionalStep, Online, Step};
use crate::problem::FractionalSimplifiedSmoothedConvexOptimization;
use crate::result::{Error, Result};
use crate::schedule::FractionalSchedule;
use crate::utils::assert;
use crate::PRECISION;
use bacon_sci::differentiate::second_derivative;
use bacon_sci::integrate::integrate;
use std::sync::Arc;

/// Probability distribution over the number of servers.
pub type Memory<'a> = Arc<dyn Fn(f64) -> f64 + 'a>;

/// Probabilistic Algorithm
pub fn probabilistic<'a>(
    o: &'a Online<FractionalSimplifiedSmoothedConvexOptimization<'a>>,
    xs: &mut FractionalSchedule,
    ps: &mut Vec<Memory<'a>>,
    _: &(),
) -> Result<FractionalStep<Memory<'a>>> {
    assert(o.w == 0, Error::UnsupportedPredictionWindow)?;
    assert(o.p.d == 1, Error::UnsupportedProblemDimension)?;

    let t = xs.t_end() + 1;
    let prev_p = if ps.is_empty() {
        Arc::new(|j| if j == 0. { 1. } else { 0. })
    } else {
        ps[ps.len() - 1].clone()
    };

    let x_m = find_minimizer_of_hitting_cost(
        t,
        &o.p.hitting_cost,
        &vec![(0., o.p.bounds[0])],
    )?
    .0[0];
    let x_r = find_right_bound(o, t, &prev_p, x_m)?;
    let x_l = find_left_bound(o, t, &prev_p, x_m)?;

    let p: Arc<dyn Fn(f64) -> f64> = Arc::new(move |j| {
        if j >= x_l && j <= x_r {
            prev_p(j)
                + second_derivative(
                    |j: f64| (o.p.hitting_cost)(t, Config::single(j)).unwrap(),
                    j,
                    PRECISION,
                ) / 2.
        } else {
            0.
        }
    });

    let x = expected_value(&p, x_l, x_r)?;
    Ok(Step(Config::single(x), Some(p)))
}

/// Determines `x_r` with a convex optimization.
fn find_right_bound(
    o: &Online<FractionalSimplifiedSmoothedConvexOptimization<'_>>,
    t: i32,
    prev_p: &Memory<'_>,
    x_m: f64,
) -> Result<f64> {
    let bounds = vec![(0., o.p.bounds[0])];
    let f = |xs: &[f64]| xs[0];
    let g = Arc::new(|xs: &[f64]| -> f64 {
        let l = integrate(
            x_m,
            xs[0],
            |j: f64| {
                second_derivative(
                    |j: f64| (o.p.hitting_cost)(t, Config::single(j)).unwrap(),
                    j,
                    PRECISION,
                )
            },
            PRECISION,
        )
        .unwrap();
        let r = integrate(xs[0], f64::INFINITY, |j: f64| prev_p(j), PRECISION)
            .unwrap();
        l / 2. - r
    });
    let init = vec![x_m];

    let (xs, _) = maximize(f, &bounds, Some(init), vec![], vec![g])?;
    Ok(xs[0])
}

/// Determines `x_l` with a convex optimization.
fn find_left_bound(
    o: &Online<FractionalSimplifiedSmoothedConvexOptimization<'_>>,
    t: i32,
    prev_p: &Memory<'_>,
    x_m: f64,
) -> Result<f64> {
    let bounds = vec![(0., o.p.bounds[0])];
    let f = |xs: &[f64]| xs[0];
    let g = Arc::new(|xs: &[f64]| -> f64 {
        let l = integrate(
            xs[0],
            x_m,
            |j: f64| {
                second_derivative(
                    |j: f64| (o.p.hitting_cost)(t, Config::single(j)).unwrap(),
                    j,
                    PRECISION,
                )
            },
            PRECISION,
        )
        .unwrap();
        let r =
            integrate(f64::NEG_INFINITY, xs[0], |j: f64| prev_p(j), PRECISION)
                .unwrap();
        l / 2. - r
    });
    let init = vec![x_m];

    let (xs, _) = minimize(f, &bounds, Some(init), vec![], vec![g])?;
    Ok(xs[0])
}

fn expected_value(p: &Memory, a: f64, b: f64) -> Result<f64> {
    integrate(a, b, |j: f64| j * p(j), PRECISION).map_err(Error::Integration)
}
