#![allow(clippy::module_inception)]
#![allow(clippy::ptr_arg)]

/// Precision used for numeric computations.
static PRECISION: f64 = 1e-6;

pub mod algorithms;
pub mod analysis;
pub mod convert;
pub mod cost;
pub mod online;
pub mod problem;
pub mod result;
pub mod schedule;
pub mod utils;
pub mod verifiers;
