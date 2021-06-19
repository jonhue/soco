use crate::config::Config;
use crate::convex_optimization::find_minimizer;
use crate::objective::Objective;
use crate::problem::FractionalSmoothedConvexOptimization;
use crate::result::Result;
use crate::schedule::{FractionalSchedule, Schedule};

/// Convex Optimization
pub fn co(
    p: &FractionalSmoothedConvexOptimization<'_>,
) -> Result<FractionalSchedule> {
    let (lower, upper): (Vec<_>, Vec<_>) = p.bounds.iter().cloned().unzip();
    let extended_lower = Schedule::build_raw(p.t_end, &Config::new(lower));
    let extended_upper = Schedule::build_raw(p.t_end, &Config::new(upper));
    let bounds = extended_lower
        .into_iter()
        .zip(extended_upper.into_iter())
        .collect();
    let f = |raw_xs: &[f64]| {
        let xs = Schedule::from_raw(p.d, p.t_end, raw_xs);
        p.objective_function(&xs).unwrap()
    };

    let (raw_xs, _) = find_minimizer(f, &bounds)?;
    Ok(Schedule::from_raw(p.d, p.t_end, &raw_xs))
}
