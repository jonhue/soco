use crate::config::Config;
use crate::objective::Objective;
use crate::problem::FractionalSmoothedConvexOptimization;
use crate::result::Result;
use crate::schedule::{FractionalSchedule, Schedule};
use crate::PRECISION;
use nlopt::{Algorithm, Nlopt, Target};

/// Convex Optimization
pub fn co(
    p: &FractionalSmoothedConvexOptimization<'_>,
) -> Result<FractionalSchedule> {
    let (lower, upper): (Vec<_>, Vec<_>) = p.bounds.iter().cloned().unzip();
    let extended_lower = Schedule::build_raw(p.t_end, &Config::new(lower));
    let extended_upper = Schedule::build_raw(p.t_end, &Config::new(upper));

    let objective_function =
        |raw_xs: &[f64], _: Option<&mut [f64]>, _: &mut ()| {
            let xs = Schedule::from_raw(p.d, p.t_end, raw_xs);
            p.objective_function(&xs).unwrap()
        };
    let mut raw_xs = extended_lower.clone();

    let mut opt = Nlopt::new(
        Algorithm::Bobyqa,
        raw_xs.len(),
        objective_function,
        Target::Minimize,
        (),
    );
    opt.set_lower_bounds(&extended_lower)?;
    opt.set_upper_bounds(&extended_upper)?;
    opt.set_xtol_rel(PRECISION)?;

    opt.optimize(&mut raw_xs)?;
    Ok(Schedule::from_raw(p.d, p.t_end, &raw_xs))
}
