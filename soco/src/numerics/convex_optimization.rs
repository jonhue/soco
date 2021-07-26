//! Convex optimization.

use crate::config::{Config, FractionalConfig};
use crate::cost::CostFn;
use crate::numerics::{ApplicablePrecision, TOLERANCE};
use crate::result::{Failure, Result};
use log::warn;
use nlopt::{Algorithm, Nlopt, Target};
use noisy_float::prelude::*;
use std::sync::Arc;

static MAX_ITERATIONS: u32 = 1_000;

/// Optimization direction.
#[derive(Clone, Copy)]
enum Direction {
    Minimize,
    Maximize,
}

/// Convex inequality constraint. The used algorithms do not support equality
/// constraints very well, and thus they are not supported by this interface.
#[derive(Clone)]
pub struct Constraint<'a, D> {
    /// Cached argument.
    pub data: D,
    /// Constraint.
    pub g: Arc<dyn Fn(&[f64], &mut D) -> N64 + 'a>,
}

/// Optimization result comprised of argmin and min.
type OptimizationResult = (Vec<f64>, N64);

/// Determines the minimizer of `hitting_cost` at time `t` with bounds `bounds`.
pub fn find_minimizer_of_hitting_cost(
    t: i32,
    hitting_cost: &CostFn<'_, FractionalConfig>,
    bounds: &Vec<(f64, f64)>,
) -> Result<OptimizationResult> {
    let f = |x: &[f64]| {
        hitting_cost.call_certain_within_bounds(
            t,
            Config::new(x.to_vec()),
            bounds,
        )
    };
    find_minimizer(f, bounds)
}

/// Determines the minimizer of a convex function `f` with bounds `bounds`.
pub fn find_minimizer(
    f: impl Fn(&[f64]) -> N64,
    bounds: &Vec<(f64, f64)>,
) -> Result<OptimizationResult> {
    minimize(f, bounds, vec![], Vec::<Constraint<()>>::new())
}

/// Determines the minimizer of a convex function `f` in `d` dimensions with `constraints`.
pub fn find_unbounded_minimizer<D>(
    f: impl Fn(&[f64]) -> N64,
    d: i32,
    constraints: Vec<Constraint<D>>,
) -> Result<OptimizationResult>
where
    D: Clone,
{
    let (bounds, init) = build_empty_bounds(d);
    minimize(f, &bounds, vec![init], constraints)
}

/// Determines the maximizer of a convex function `f` in `d` dimensions with `constraints`.
pub fn find_unbounded_maximizer<D>(
    f: impl Fn(&[f64]) -> N64,
    d: i32,
    constraints: Vec<Constraint<D>>,
) -> Result<OptimizationResult>
where
    D: Clone,
{
    let (bounds, init) = build_empty_bounds(d);
    maximize(f, &bounds, vec![init], constraints)
}

pub fn minimize<D>(
    f: impl Fn(&[f64]) -> N64,
    bounds: &Vec<(f64, f64)>,
    strategies: Vec<Vec<f64>>,
    constraints: Vec<Constraint<D>>,
) -> Result<OptimizationResult>
where
    D: Clone,
{
    evaluate_strategies(
        Direction::Minimize,
        &f,
        bounds,
        strategies,
        constraints,
    )
}

pub fn maximize<D>(
    f: impl Fn(&[f64]) -> N64,
    bounds: &Vec<(f64, f64)>,
    strategies: Vec<Vec<f64>>,
    constraints: Vec<Constraint<D>>,
) -> Result<OptimizationResult>
where
    D: Clone,
{
    evaluate_strategies(
        Direction::Maximize,
        &f,
        bounds,
        strategies,
        constraints,
    )
}

/// Applies provided stratedies one-by-one and takes the first one that produces a finite result.
fn evaluate_strategies<D>(
    dir: Direction,
    f: &impl Fn(&[f64]) -> N64,
    bounds: &Vec<(f64, f64)>,
    strategies: Vec<Vec<f64>>,
    constraints: Vec<Constraint<D>>,
) -> Result<OptimizationResult>
where
    D: Clone,
{
    let result = strategies.into_iter().find_map(|init| {
        let (x, opt) =
            optimize(dir, f, bounds, Some(init), constraints.clone()).unwrap();
        if opt.is_finite() {
            Some((x, opt))
        } else {
            None
        }
    });
    match result {
        Some(result) => Ok(result),
        None => optimize(dir, f, bounds, None, constraints),
    }
}

/// Determines the optimum of a convex function `f` w.r.t some direction `dir`
/// with bounds `bounds`, and `constraints`.
/// Optimization begins at `init` (defaults to lower bounds).
fn optimize<D>(
    dir: Direction,
    f: &impl Fn(&[f64]) -> N64,
    bounds: &Vec<(f64, f64)>,
    init: Option<Vec<f64>>,
    constraints: Vec<Constraint<D>>,
) -> Result<OptimizationResult> {
    let d = bounds.len();
    let (lower, upper): (Vec<_>, Vec<_>) = bounds.iter().cloned().unzip();

    let objective = |xs: &[f64], _: Option<&mut [f64]>, _: &mut ()| {
        evaluate(xs, &mut (), &|x, _| f(x))
    };
    let mut x = match init {
        // we use the upper bound of the decision space as for mose cost functions
        // this appears to be the most conservative estimate for a region of the
        // decision space where the hitting cost is not infinity
        None => {
            assert!(upper.iter().all(|&b| b < f64::INFINITY), "Initial guess must be set explicitly when optimization problem does not have fixed upper bounds.");
            upper.clone()
        }
        Some(x) => {
            assert!(x.len() == bounds.len());
            x
        }
    };

    let mut solver = Nlopt::new(
        choose_algorithm(constraints.len()),
        d,
        objective,
        Target::from(dir),
        (),
    );
    solver.set_lower_bounds(&lower)?;
    solver.set_upper_bounds(&upper)?;
    solver.set_xtol_rel(TOLERANCE)?;

    // stop evaluation when solver appears to hit a dead end, this may happen when all function evaluations return infinity.
    solver.set_maxeval(MAX_ITERATIONS)?;

    for Constraint { g, data } in constraints {
        solver.add_inequality_constraint(
            |xs: &[f64], _: Option<&mut [f64]>, data: &mut D| {
                evaluate(xs, data, &|x, data| g(x, data))
            },
            data,
            TOLERANCE,
        )?;
    }

    let opt = match solver.optimize(&mut x) {
        Ok((state, opt)) => Ok(match state {
            nlopt::SuccessState::MaxEvalReached
            | nlopt::SuccessState::MaxTimeReached => {
                warn!("Convex optimization timed out. Assuming solution to be infinity.");
                f64::INFINITY
            }
            _ => opt,
        }),
        Err((state, opt)) => match state {
            nlopt::FailState::RoundoffLimited => {
                warn!("Warning: NLOpt terminated with a roundoff error.");
                Ok(opt)
            }
            _ => Err(Failure::NlOpt(state)),
        },
    }?;
    Ok((x.apply_precision(), n64(opt)))
}

fn choose_algorithm(constraints: usize) -> Algorithm {
    // both Cobyla and Bobyqa are algorithms for derivative-free local optimization
    if constraints > 0 {
        Algorithm::Cobyla
    } else {
        // Bobyqa does not support (in-)equality constraints
        Algorithm::Bobyqa
    }
}

impl From<Direction> for Target {
    fn from(dir: Direction) -> Self {
        match dir {
            Direction::Minimize => Target::Minimize,
            Direction::Maximize => Target::Maximize,
        }
    }
}

/// Returns empty bounds and init vector.
fn build_empty_bounds(d: i32) -> (Vec<(f64, f64)>, Vec<f64>) {
    (
        vec![(f64::NEG_INFINITY, f64::INFINITY); d as usize],
        vec![0.; d as usize],
    )
}

/// It appears that NLOpt sometimes produces NaN values for no reason.
/// This is to ensure that NaN values are not chosen.
fn evaluate<D>(
    xs: &[f64],
    data: &mut D,
    f: &impl Fn(&[f64], &mut D) -> N64,
) -> f64 {
    if xs.iter().any(|&x| x.is_nan()) {
        f64::NAN
    } else {
        f(xs, data).raw()
    }
}
