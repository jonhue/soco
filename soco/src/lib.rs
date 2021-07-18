#![allow(clippy::many_single_char_names)]
#![allow(clippy::module_inception)]
#![allow(clippy::ptr_arg)]

use std::collections::HashMap;

#[allow(unused_imports)]
#[macro_use]
extern crate approx;

#[macro_use]
extern crate derivative;

pub mod algorithms;
pub mod breakpoints;
pub mod config;
pub mod convert;
pub mod cost;
pub mod model;
pub mod norm;
pub mod objective;
pub mod problem;
pub mod result;
pub mod schedule;
pub mod streaming;
pub mod verifiers;

mod numerics;
mod utils;
mod value;
mod vec_wrapper;

/// Constructs a hash map from a slice.
pub fn hash_map<K, V>(slice: &[(K, V)]) -> HashMap<K, V>
where
    K: Clone + Eq + std::hash::Hash + PartialEq,
    V: Clone,
{
    slice.iter().cloned().collect()
}
