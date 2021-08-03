//! Online Balanced Descent

use crate::config::FractionalConfig;
use std::sync::Arc;

/// Maximum `l` for bisection is calculated as `MAX_L_FACTOR * min_hitting_cost`.
static MAX_L_FACTOR: f64 = 1_000.;

pub mod dual;
pub mod greedy;
pub mod meta;
pub mod primal;
pub mod regularized;

pub type DistanceGeneratingFn =
    Arc<dyn Fn(FractionalConfig) -> f64 + Send + Sync>;
