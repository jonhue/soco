//! Optimization utils.

use crate::config::{Config, FractionalConfig};
use crate::cost::CostFn;
use crate::result::{Error, Result};
use crate::utils::assert;
use crate::PRECISION;
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
    let f = |x: &[f64]| hitting_cost(t, Config::new(x.to_vec())).unwrap();
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
    let bounds = build_empty_bounds(d);
    minimize(
        f,
        &bounds,
        None,
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
        None => lower.clone(),
        Some(x) => {
            assert(x.len() == bounds.len(), Error::DimensionInconsistent)?;
            x
        }
    };

    let mut solver =
        Nlopt::new(Algorithm::Bobyqa, d, objective, Target::from(dir), ());
    solver.set_lower_bounds(&lower)?;
    solver.set_upper_bounds(&upper)?;
    solver.set_xtol_rel(PRECISION)?;

    for g in inequality_constraints {
        solver.add_inequality_constraint(
            |xs: &[f64], _: Option<&mut [f64]>, _: &mut ()| g(xs),
            (),
            PRECISION,
        )?;
    }
    for g in equality_constraints {
        solver.add_equality_constraint(
            |xs: &[f64], _: Option<&mut [f64]>, _: &mut ()| g(xs),
            (),
            PRECISION,
        )?;
    }

    let opt = solver.optimize(&mut x)?.1;
    Ok((x, opt))
}

impl From<Direction> for Target {
    fn from(dir: Direction) -> Self {
        match dir {
            Direction::Minimize => Target::Minimize,
            Direction::Maximize => Target::Maximize,
        }
    }
}

fn build_empty_bounds(d: i32) -> Vec<(f64, f64)> {
    vec![(f64::NEG_INFINITY, f64::INFINITY); d as usize]
}
