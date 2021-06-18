//! Optimization utils.

use crate::config::{Config, FractionalConfig};
use crate::cost::CostFn;
use crate::result::Result;
use crate::PRECISION;
use nlopt::{Algorithm, Nlopt, Target};

/// Determines the minimizer of `hitting_cost` at time `t` with bounds `bounds`
pub fn find_minimizer_of_hitting_cost(
    t: i32,
    hitting_cost: &CostFn<'_, FractionalConfig>,
    bounds: &Vec<(f64, f64)>,
) -> Result<FractionalConfig> {
    let f = |x: &[f64]| hitting_cost(t, Config::new(x.to_vec())).unwrap();
    find_minimizer(f, bounds)
}

/// Determines the minimizer of a convex function `f` with bounds `bounds`
pub fn find_minimizer(
    f: impl Fn(&[f64]) -> f64,
    bounds: &Vec<(f64, f64)>,
) -> Result<FractionalConfig> {
    let d = bounds.len();
    let (lower, upper): (Vec<_>, Vec<_>) = bounds.iter().cloned().unzip();

    let objective = |x: &[f64], _: Option<&mut [f64]>, _: &mut ()| f(x);
    let mut x = lower.clone();

    let mut opt =
        Nlopt::new(Algorithm::Bobyqa, d, objective, Target::Minimize, ());
    opt.set_lower_bounds(&lower)?;
    opt.set_upper_bounds(&upper)?;
    opt.set_xtol_rel(PRECISION)?;

    opt.optimize(&mut x)?;
    Ok(Config::new(x))
}
