//! Online Balanced Descent

use crate::config::FractionalConfig;
use std::sync::Arc;

pub mod dual;
pub mod greedy;
pub mod meta;
pub mod primal;
pub mod regularized;

pub type DistanceGeneratingFn =
    Arc<dyn Fn(FractionalConfig) -> f64 + Send + Sync>;
