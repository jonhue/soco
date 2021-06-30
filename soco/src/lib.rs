#![allow(clippy::many_single_char_names)]
#![allow(clippy::module_inception)]
#![allow(clippy::ptr_arg)]

#[allow(unused_imports)]
#[macro_use]
extern crate approx;

pub mod algorithms;
pub mod breakpoints;
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

mod numerics;
mod utils;
mod value;
mod vec_wrapper;
