use crate::algorithms::capacity_provisioning::Bounded;
use crate::algorithms::online::Step;
use crate::config::Config;
use crate::model::{ModelOutputFailure, ModelOutputSuccess};
use crate::numerics::PRECISION;
use crate::problem::{Online, Problem};
use crate::result::{Failure, Result};
use crate::schedule::Schedule;
use crate::utils::{assert, project};
use crate::value::Value;
use num::{NumCast, ToPrimitive};
use pyo3::prelude::*;
use serde_derive::{Deserialize, Serialize};

/// Lower and upper bound from some time `t`.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct BoundsMemory<T> {
    lower: T,
    upper: T,
}
impl<T> IntoPy<PyObject> for BoundsMemory<T>
where
    T: IntoPy<PyObject>,
{
    fn into_py(self, py: Python) -> PyObject {
        (self.lower, self.upper).into_py(py)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Memory<T> {
    /// Lower and upper bounds from times `t` (in order).
    bounds: Vec<BoundsMemory<T>>,
}
impl<T> Default for Memory<T> {
    fn default() -> Self {
        Memory { bounds: vec![] }
    }
}
impl<T> IntoPy<PyObject> for Memory<T>
where
    T: IntoPy<PyObject>,
{
    fn into_py(self, py: Python) -> PyObject {
        self.bounds.into_py(py)
    }
}

/// Lazy Capacity Provisioning
pub fn lcp<'a, T, P, C, D>(
    o: Online<P>,
    t: i32,
    xs: &Schedule<T>,
    Memory { mut bounds }: Memory<T>,
    _: (),
) -> Result<Step<T, Memory<T>>>
where
    T: Value<'a>,
    P: Bounded<T> + Problem<T, C, D>,
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    assert(o.p.d() == 1, Failure::UnsupportedProblemDimension(o.p.d()))?;
    assert(
        t - 1 == bounds.len() as i32,
        Failure::OnlineOutOfDateMemory {
            previous_time_slots: t - 1,
            memory_entries: bounds.len() as i32,
        },
    )?;

    let (t_start, x_start) = find_initial_time(&bounds);

    let i = xs.now_with_default(Config::single(NumCast::from(0).unwrap()))[0];
    let lower = o.p.find_lower_bound(o.w, o.p.t_end(), t_start, x_start)?;
    let upper = o.p.find_upper_bound(o.w, o.p.t_end(), t_start, x_start)?;
    let j = project(i, lower, upper);

    bounds.push(BoundsMemory { lower, upper });
    let m = Memory { bounds };

    Ok(Step(Config::single(j), Some(m)))
}

/// Finds a valid reference time and initial condition to base the optimization
/// on (alternatively to time `0`).
fn find_initial_time<'a, T>(bounds: &Vec<BoundsMemory<T>>) -> (i32, T)
where
    T: Value<'a>,
{
    for t in (2..=bounds.len() as i32).rev() {
        let prev_bound = &bounds[t as usize - 2];
        let bound = &bounds[t as usize - 1];
        if is_valid_initial_time(prev_bound, bound) {
            // this should always be true, however, it may be false due to numerical inaccuracies
            if ToPrimitive::to_f64(&(prev_bound.lower - prev_bound.upper))
                .unwrap()
                .abs()
                < PRECISION
            {
                return (t - 1, prev_bound.upper);
            }
        }
    }

    (0, NumCast::from(0).unwrap())
}

/// Returns `true` if the time `t` with current bounds `m` and previous bounds
/// (at `t - 1`) `prev_m` may be used as reference time.
fn is_valid_initial_time<'a, T>(
    BoundsMemory {
        lower: prev_lower,
        upper: prev_upper,
    }: &BoundsMemory<T>,
    BoundsMemory { lower, upper }: &BoundsMemory<T>,
) -> bool
where
    T: Value<'a>,
{
    upper < prev_upper || lower > prev_lower
}
