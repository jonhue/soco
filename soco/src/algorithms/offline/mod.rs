//! Offline Algorithms.

pub mod multi_dimensional;
pub mod uni_dimensional;

mod graph_search;

use crate::algorithms::Options;
use crate::problem::Problem;
use crate::result::Result;
use crate::schedule::Schedule;

/// Implementation of an offline algorithm.
///
/// * `T` - Result.
/// * `P` - Problem.
/// * `O` - Options.
///
/// Receives the arguments:
/// * `p` - Problem instance.
/// * `options` - Algorithm options.
/// * `inverted` - Whether to computed inverted movement costs.
pub trait OfflineAlgorithm<T, R, P, O>: Fn(P, O, bool) -> Result<R>
where
    R: OfflineResult<T>,
    P: Problem,
    O: Options<P>,
{
    fn solve(&self, p: P, options: O, inverted: bool) -> Result<R> {
        self(p, options, inverted)
    }
}
impl<T, R, P, O, F> OfflineAlgorithm<T, R, P, O> for F
where
    R: OfflineResult<T>,
    P: Problem,
    O: Options<P>,
    F: Fn(P, O, bool) -> Result<R>,
{
}

pub trait OfflineAlgorithmWithDefaultOptions<T, R, P, O>:
    OfflineAlgorithm<T, R, P, O>
where
    R: OfflineResult<T>,
    P: Problem,
    O: Options<P>,
{
    fn solve_with_default_options(&self, p: P, inverted: bool) -> Result<R> {
        let options = O::default(&p);
        OfflineAlgorithm::solve(self, p, options, inverted)
    }
}
impl<T, R, P, O, F> OfflineAlgorithmWithDefaultOptions<T, R, P, O> for F
where
    R: OfflineResult<T>,
    P: Problem,
    O: Options<P>,
    F: Fn(P, O, bool) -> Result<R>,
{
}

/// Result of an offline algorithm.
pub trait OfflineResult<T> {
    fn xs(self) -> Schedule<T>;
}

/// Result of an offline algorithm which only returns the obtained schedule.
pub struct PureOfflineResult<T> {
    xs: Schedule<T>,
}
impl<T> OfflineResult<T> for PureOfflineResult<T> {
    fn xs(self) -> Schedule<T> {
        self.xs
    }
}
