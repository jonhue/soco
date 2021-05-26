//! Optimization utils.

use nlopt::{Algorithm, Nlopt, Target};

use crate::config::Config;
use crate::cost::CostFn;
use crate::result::Result;
use crate::PRECISION;

/// Determines the minimizer of `f` at time `t` with bounds `bounds`
pub fn find_minimizer(
    t: i32,
    f: &CostFn<'_, Config<f64>>,
    bounds: &Vec<(f64, f64)>,
) -> Result<Config<f64>> {
    let d = bounds.len();
    let (lower, upper): (Vec<_>, Vec<_>) = bounds.iter().cloned().unzip();

    let objective_function =
        |x: &[f64], _: Option<&mut [f64]>, _: &mut ()| -> f64 {
            f(t, Config::new(x.to_vec())).unwrap()
        };
    let mut x = lower.clone();

    let mut opt = Nlopt::new(
        Algorithm::Bobyqa,
        d,
        objective_function,
        Target::Minimize,
        (),
    );
    opt.set_lower_bounds(&lower)?;
    opt.set_upper_bounds(&upper)?;
    opt.set_xtol_rel(PRECISION)?;

    opt.optimize(&mut x)?;
    Ok(Config::new(x))
}
