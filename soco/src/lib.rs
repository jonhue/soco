#![allow(clippy::many_single_char_names)]
#![allow(clippy::module_inception)]
#![allow(clippy::ptr_arg)]

/// Precision used for numeric computations.
static PRECISION: f64 = 1e-6;
/// Tolerance used for numeric computations.
static TOLERANCE: f64 = PRECISION / 10.;

pub mod algorithms;
pub mod config;
pub mod convert;
pub mod cost;
pub mod norm;
pub mod objective;
pub mod online;
pub mod problem;
pub mod result;
pub mod schedule;
pub mod verifiers;

mod convex_optimization;
mod quadrature;
mod utils;
mod value;
mod vec_wrapper;
