use super::next;
use crate::algorithms::online::{FractionalStep, Step};
use crate::config::Config;
use crate::model::{ModelOutputFailure, ModelOutputSuccess};
use crate::problem::{FractionalSimplifiedSmoothedConvexOptimization, Online};
use crate::result::Result;
use crate::schedule::FractionalSchedule;

/// Receding Horizon Control
pub fn rhc<C, D>(
    o: Online<FractionalSimplifiedSmoothedConvexOptimization<C, D>>,
    t: i32,
    xs: &FractionalSchedule,
    _: (),
    _: (),
) -> Result<FractionalStep<()>>
where
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    let prev_x = xs.now_with_default(Config::repeat(0., o.p.d));
    let (_, x) = next(o.w + 1, o, t, prev_x);
    Ok(Step(x, None))
}
