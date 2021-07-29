//! Convex optimization.

use crate::config::{Config, FractionalConfig};
use crate::cost::CostFn;
use crate::model::{ModelOutputFailure, ModelOutputSuccess};
use crate::numerics::{ApplicablePrecision, TOLERANCE};
use log::warn;
use nlopt::{Algorithm, Nlopt, Target};
use noisy_float::prelude::*;
use std::sync::Arc;

static MAX_ITERATIONS_PER_DIM: u32 = 1_000;

/// Optimization direction.
#[derive(Clone, Copy)]
enum Direction {
    Minimize,
    Maximize,
}

type ObjectiveFn<'a, D> = Arc<dyn Fn(&[f64], &mut D) -> N64 + 'a>;

/// Wrapper around objectives.
#[derive(Clone)]
pub struct WrappedObjective<'a, D> {
    /// Cached argument.
    data: D,
    /// Objective.
    f: ObjectiveFn<'a, D>,
}
impl<'a, D> WrappedObjective<'a, D> {
    pub fn new(data: D, f: impl Fn(&[f64], &mut D) -> N64 + 'a) -> Self {
        Self {
            data,
            f: Arc::new(f),
        }
    }
}

/// Optimization result comprised of argmin and min.
type OptimizationResult = (Vec<f64>, N64);

/// Determines the minimizer of `hitting_cost` at time `t` with bounds `bounds`.
pub fn find_minimizer_of_hitting_cost<C, D>(
    t: i32,
    hitting_cost: CostFn<'_, FractionalConfig, C, D>,
    bounds: Vec<(f64, f64)>,
) -> OptimizationResult
where
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    let objective = WrappedObjective::new(hitting_cost, |x, hitting_cost| {
        hitting_cost.call_certain(t, Config::new(x.to_vec())).cost
    });
    find_minimizer(objective, bounds)
}

/// Determines the minimizer of a convex function `f` with bounds `bounds`.
pub fn find_minimizer<C>(
    objective: WrappedObjective<C>,
    bounds: Vec<(f64, f64)>,
) -> OptimizationResult
where
    C: Clone,
{
    minimize(objective, bounds, None, Vec::<WrappedObjective<()>>::new())
}

/// Determines the minimizer of a convex function `f` in `d` dimensions with `constraints`.
pub fn find_unbounded_minimizer<C, D>(
    objective: WrappedObjective<C>,
    d: i32,
    constraints: Vec<WrappedObjective<D>>,
) -> OptimizationResult
where
    C: Clone,
    D: Clone,
{
    let (bounds, init) = build_empty_bounds(d);
    minimize(objective, bounds, Some(init), constraints)
}

/// Determines the maximizer of a convex function `f` in `d` dimensions with `constraints`.
pub fn find_unbounded_maximizer<C, D>(
    objective: WrappedObjective<C>,
    d: i32,
    constraints: Vec<WrappedObjective<D>>,
) -> OptimizationResult
where
    C: Clone,
    D: Clone,
{
    let (bounds, init) = build_empty_bounds(d);
    maximize(objective, bounds, Some(init), constraints)
}

pub fn minimize<C, D>(
    objective: WrappedObjective<C>,
    bounds: Vec<(f64, f64)>,
    init: Option<Vec<f64>>,
    constraints: Vec<WrappedObjective<D>>,
) -> OptimizationResult
where
    C: Clone,
    D: Clone,
{
    optimize(Direction::Minimize, objective, bounds, init, constraints)
}

pub fn maximize<C, D>(
    objective: WrappedObjective<C>,
    bounds: Vec<(f64, f64)>,
    init: Option<Vec<f64>>,
    constraints: Vec<WrappedObjective<D>>,
) -> OptimizationResult
where
    C: Clone,
    D: Clone,
{
    optimize(Direction::Maximize, objective, bounds, init, constraints)
}

/// Determines the optimum of a convex function `f` w.r.t some direction `dir`
/// with bounds `bounds`, and `constraints`.
/// Optimization begins at `init` (defaults to lower bounds).
///
/// The used algorithms do not support equality constraints very well, and thus
/// they are not supported by this interface.
fn optimize<C, D>(
    dir: Direction,
    objective: WrappedObjective<C>,
    bounds: Vec<(f64, f64)>,
    init: Option<Vec<f64>>,
    constraints: Vec<WrappedObjective<D>>,
) -> OptimizationResult {
    let d = bounds.len();
    let (lower, upper): (Vec<_>, Vec<_>) = bounds.into_iter().unzip();

    let WrappedObjective { data, f } = objective;
    let solver_objective = |xs: &[f64], _: Option<&mut [f64]>, data: &mut C| {
        evaluate(xs, data, &f)
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
            assert!(x.len() == d);
            x
        }
    };

    let mut solver = Nlopt::new(
        choose_algorithm(constraints.len()),
        d,
        solver_objective,
        Target::from(dir),
        data,
    );
    solver.set_lower_bounds(&lower).unwrap();
    solver.set_upper_bounds(&upper).unwrap();
    solver.set_xtol_abs1(TOLERANCE).unwrap();
    solver.set_xtol_rel(TOLERANCE).unwrap();

    // stop evaluation when solver appears to hit a dead end, this may happen when all function evaluations return infinity.
    solver
        .set_maxeval(d as u32 * MAX_ITERATIONS_PER_DIM)
        .unwrap();

    for WrappedObjective { f, data } in constraints {
        solver
            .add_inequality_constraint(
                |xs: &[f64], _: Option<&mut [f64]>, data: &mut D| {
                    evaluate(xs, data, &f)
                },
                data,
                TOLERANCE,
            )
            .unwrap();
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
            _ => Err((state, opt)),
        },
    }.unwrap();

    (x.apply_precision(), n64(opt))
}

fn choose_algorithm(constraints: usize) -> Algorithm {
    // We use algorithms for derivative-free local optimization
    #[allow(clippy::if_same_then_else)]
    if constraints > 0 {
        // Only Cobyla supports (in-)equality constraints
        Algorithm::Cobyla
    } else {
        // This might require some re-configuration depending on the problem at hand.
        // Viable options are `Sbplex`, `Cobyla`, (a little less often) `Praxis`,
        // and (in some few cases) `Bobyqa`.
        Algorithm::Sbplx
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
fn evaluate<D>(xs: &[f64], data: &mut D, f: &ObjectiveFn<D>) -> f64 {
    if xs.iter().any(|&x| x.is_nan()) {
        f64::NAN
    } else {
        f(xs, data).raw()
    }
}
