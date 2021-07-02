//! Offline Algorithms.

pub mod multi_dimensional;
pub mod uni_dimensional;

use crate::algorithms::Options;
use crate::problem::Problem;
use crate::result::Result;

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
pub trait OfflineAlgorithm<T, P, O>: Fn(P, O, bool) -> Result<T>
where
    P: Problem,
    O: Options<P>,
{
    fn solve(&self, p: P, options: O, inverted: bool) -> Result<T> {
        self(p, options, inverted)
    }
}
impl<T, P, O, F> OfflineAlgorithm<T, P, O> for F
where
    P: Problem,
    O: Options<P>,
    F: Fn(P, O, bool) -> Result<T>,
{
}

pub trait OfflineAlgorithmWithDefaultOptions<T, P, O>:
    OfflineAlgorithm<T, P, O>
where
    P: Problem,
    O: Options<P>,
{
    fn solve_with_default_options(&self, p: P, inverted: bool) -> Result<T> {
        let options = O::default(&p);
        OfflineAlgorithm::solve(self, p, options, inverted)
    }
}
impl<T, P, O, F> OfflineAlgorithmWithDefaultOptions<T, P, O> for F
where
    P: Problem,
    O: Options<P>,
    F: Fn(P, O, bool) -> Result<T>,
{
}
