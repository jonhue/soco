use crate::algorithms::capacity_provisioning::{Bounded, BoundsMemory};
use crate::algorithms::offline::{OfflineOptions, OfflineResult};
use crate::config::Config;
use crate::model::{ModelOutputFailure, ModelOutputSuccess};
use crate::problem::FractionalSimplifiedSmoothedConvexOptimization;
use crate::result::{Failure, Result};
use crate::schedule::{FractionalSchedule, Schedule};
use crate::utils::{assert, project};

/// Schedule along with lower and upper bounds.
pub struct WithBounds {
    /// Schedule.
    pub xs: FractionalSchedule,
    /// Lower and upper bounds.
    pub bounds: Vec<BoundsMemory<f64>>,
}
impl OfflineResult<f64> for WithBounds {
    fn xs(self) -> FractionalSchedule {
        self.xs
    }
}

/// Backward-Recurrent Capacity Provisioning
pub fn brcp<C, D>(
    p: FractionalSimplifiedSmoothedConvexOptimization<'_, C, D>,
    _: (),
    OfflineOptions { inverted, alpha, l }: OfflineOptions,
) -> Result<WithBounds>
where
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    assert(!inverted, Failure::UnsupportedInvertedCost)?;
    assert(l.is_none(), Failure::UnsupportedLConstrainedMovement)?;
    assert(p.d == 1, Failure::UnsupportedProblemDimension(p.d))?;

    let mut xs = Schedule::empty();

    let mut bounds = Vec::new();
    let mut x = 0.;
    let mut bound;
    for t in (1..=p.t_end).rev() {
        (x, bound) = next(&p, alpha, t, x)?;
        xs.shift(Config::single(x));
        bounds.insert(0, bound);
    }

    Ok(WithBounds { xs, bounds })
}

fn next<C, D>(
    p: &FractionalSimplifiedSmoothedConvexOptimization<'_, C, D>,
    alpha: f64,
    t: i32,
    x: f64,
) -> Result<(f64, BoundsMemory<f64>)>
where
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    let lower = p.find_alpha_unfair_lower_bound(alpha, 0, t, 0, 0.)?;
    let upper = p.find_alpha_unfair_upper_bound(alpha, 0, t, 0, 0.)?;

    Ok((project(x, lower, upper), BoundsMemory { lower, upper }))
}
