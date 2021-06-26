use crate::algorithms::capacity_provisioning::Bounded;
use crate::config::Config;
use crate::online::{Online, Step};
use crate::problem::Problem;
use crate::result::{Error, Result};
use crate::schedule::Schedule;
use crate::utils::{assert, project};
use crate::value::Value;
use num::NumCast;

/// Last two lower and upper bound from some time `t` (in order).
#[derive(Clone, Debug)]
pub struct Memory<T>
where
    T: Value,
{
    pub lower: (Option<T>, T),
    pub upper: (Option<T>, T),
}

/// Lazy Capacity Provisioning
pub fn lcp<T, U>(
    o: &Online<U>,
    xs: &mut Schedule<T>,
    ms: &mut Vec<Memory<T>>,
    _: &(),
) -> Result<Step<T, Memory<T>>>
where
    T: Value,
    U: Bounded<T> + Problem,
{
    assert(o.p.d() == 1, Error::UnsupportedProblemDimension)?;

    let (t_start, x_start) = find_initial_time(o.p.t_end(), ms);

    let i = if xs.is_empty() {
        NumCast::from(0).unwrap()
    } else {
        xs.now()[0]
    };
    let l = o.p.find_lower_bound(o.p.t_end(), t_start, x_start)?;
    let u = o.p.find_upper_bound(o.p.t_end(), t_start, x_start)?;
    let j = project(i, l, u);
    Ok(Step(Config::single(j), Some(new_memory(ms, l, u))))
}

/// Finds a valid reference time and initial condition to base the optimization
/// on (alternatively to time `0`).
fn find_initial_time<T>(final_t: i32, ms: &Vec<Memory<T>>) -> (i32, T)
where
    T: Value,
{
    for t in (2..final_t).rev() {
        let m = &ms[t as usize - 1];
        if is_valid_initial_time(&m) {
            return (t, m.upper.0.unwrap());
        }
    }

    (0, NumCast::from(0).unwrap())
}

/// Returns `true` if the time `t` with current bounds `m` and previous bounds
/// (at `t - 1`) `prev_m` may be used as reference time.
fn is_valid_initial_time<T>(Memory { lower, upper }: &Memory<T>) -> bool
where
    T: Value,
{
    upper.1 < upper.0.unwrap() || lower.1 > lower.0.unwrap()
}

/// Shifts the memory to include new lower bound `l` and upper bound `u`.
fn new_memory<T>(ms: &Vec<Memory<T>>, l: T, u: T) -> Memory<T>
where
    T: Value,
{
    let (prev_l, prev_u) = if ms.is_empty() {
        (None, None)
    } else {
        let m = &ms[ms.len() - 1];
        (Some(m.lower.1), Some(m.upper.1))
    };
    Memory {
        lower: (prev_l, l),
        upper: (prev_u, u),
    }
}