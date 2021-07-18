//! Convex optimization.

use crate::config::{Config, FractionalConfig};
use crate::cost::CostFn;
use crate::numerics::{ApplicablePrecision, TOLERANCE};
use crate::result::{Failure, Result};
use nlopt::{Algorithm, Nlopt, Target};
use std::sync::Arc;

/// Optimization direction.
enum Direction {
    Minimize,
    Maximize,
}

/// Convex constraint.
pub type Constraint<'a> = Arc<dyn Fn(&[f64]) -> f64 + 'a>;

/// Optimization result comprised of argmin and min.
type OptimizationResult = (Vec<f64>, f64);

/// Determines the minimizer of `hitting_cost` at time `t` with bounds `bounds`.
pub fn find_minimizer_of_hitting_cost(
    t: i32,
    hitting_cost: &CostFn<'_, FractionalConfig>,
    bounds: &Vec<(f64, f64)>,
) -> Result<OptimizationResult> {
    let f = |x: &[f64]| hitting_cost.call(t, Config::new(x.to_vec()), bounds);
    find_minimizer(f, bounds)
}

/// Determines the minimizer of a convex function `f` with bounds `bounds`.
pub fn find_minimizer(
    f: impl Fn(&[f64]) -> f64,
    bounds: &Vec<(f64, f64)>,
) -> Result<OptimizationResult> {
    minimize(f, bounds, None, vec![], vec![])
}

/// Determines the minimizer of a convex function `f` in `d` dimensions with
/// `inequality_constraints` and `equality_constraints`.
pub fn find_unbounded_minimizer(
    f: impl Fn(&[f64]) -> f64,
    d: i32,
    inequality_constraints: Vec<Constraint>,
    equality_constraints: Vec<Constraint>,
) -> Result<OptimizationResult> {
    let (bounds, init) = build_empty_bounds(d);
    minimize(
        f,
        &bounds,
        Some(init),
        inequality_constraints,
        equality_constraints,
    )
}

/// Determines the maximizer of a convex function `f` in `d` dimensions with
/// `inequality_constraints` and `equality_constraints`.
pub fn find_unbounded_maximizer(
    f: impl Fn(&[f64]) -> f64,
    d: i32,
    inequality_constraints: Vec<Constraint>,
    equality_constraints: Vec<Constraint>,
) -> Result<OptimizationResult> {
    let (bounds, init) = build_empty_bounds(d);
    maximize(
        f,
        &bounds,
        Some(init),
        inequality_constraints,
        equality_constraints,
    )
}

pub fn minimize(
    f: impl Fn(&[f64]) -> f64,
    bounds: &Vec<(f64, f64)>,
    init: Option<Vec<f64>>,
    inequality_constraints: Vec<Constraint>,
    equality_constraints: Vec<Constraint>,
) -> Result<OptimizationResult> {
    optimize(
        Direction::Minimize,
        f,
        bounds,
        init,
        inequality_constraints,
        equality_constraints,
    )
}

pub fn maximize(
    f: impl Fn(&[f64]) -> f64,
    bounds: &Vec<(f64, f64)>,
    init: Option<Vec<f64>>,
    inequality_constraints: Vec<Constraint>,
    equality_constraints: Vec<Constraint>,
) -> Result<OptimizationResult> {
    optimize(
        Direction::Maximize,
        f,
        bounds,
        init,
        inequality_constraints,
        equality_constraints,
    )
}

/// Determines the optimum of a convex function `f` w.r.t some direction `dir`
/// with bounds `bounds`, `inequality_constraints`, and `equality_constraints`.
/// Optimization begins at `init` (defaults to lower bounds).
fn optimize(
    dir: Direction,
    f: impl Fn(&[f64]) -> f64,
    bounds: &Vec<(f64, f64)>,
    init: Option<Vec<f64>>,
    inequality_constraints: Vec<Constraint>,
    equality_constraints: Vec<Constraint>,
) -> Result<OptimizationResult> {
    let d = bounds.len();
    let (lower, upper): (Vec<_>, Vec<_>) = bounds.iter().cloned().unzip();

    let objective = |x: &[f64], _: Option<&mut [f64]>, _: &mut ()| f(x);
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
        choose_algorithm(
            inequality_constraints.len(),
            equality_constraints.len(),
        ),
        d,
        objective,
        Target::from(dir),
        (),
    );
    solver.set_lower_bounds(&lower)?;
    solver.set_upper_bounds(&upper)?;
    solver.set_xtol_rel(TOLERANCE)?;

    for g in inequality_constraints {
        solver.add_inequality_constraint(
            |xs: &[f64], _: Option<&mut [f64]>, _: &mut ()| g(xs),
            (),
            TOLERANCE,
        )?;
    }
    for g in equality_constraints {
        solver.add_equality_constraint(
            |xs: &[f64], _: Option<&mut [f64]>, _: &mut ()| g(xs),
            (),
            TOLERANCE,
        )?;
    }

    let opt = match solver.optimize(&mut x) {
        Ok((_, opt)) => Ok(opt),
        Err((state, opt)) => match state {
            nlopt::FailState::RoundoffLimited => {
                println!("Warning: NLOpt terminated with a roundoff error.");
                Ok(opt)
            }
            _ => Err(Failure::NlOpt(state)),
        },
    }?;
    Ok((x.apply_precision(), opt))
}

fn choose_algorithm(
    inequality_constraints: usize,
    equality_constraints: usize,
) -> Algorithm {
    // both Cobyla and Bobyqa are algorithms for derivative-free local optimization
    if equality_constraints > 0 || inequality_constraints > 0 {
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
