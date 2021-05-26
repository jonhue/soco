//! Mirror maps.

use std::sync::Arc;

use crate::config::Config;
use crate::norm::NormFn;
use crate::value::Value;

/// Mirror map of some norm function.
pub type MirrorMap<'a, T> = Arc<dyn Fn(&NormFn<'a, T>, T) -> f64 + 'a>;

/// Norm squared. `1`-strongly convex and `1`-Lipschitz smooth for the Euclidean norm and the Mahalanobis distance.
pub fn norm_squared<T>(norm: NormFn<'_, Config<T>>, x: Config<T>) -> f64
where
    T: Value,
{
    norm(x).powi(2) / 2.
}
