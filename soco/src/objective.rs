//! Objective function.

use crate::config::Config;
use crate::cost::{Cost, RawCost};
use crate::model::{ModelOutputFailure, ModelOutputSuccess};
use crate::problem::Problem;
use crate::result::Result;
use crate::schedule::Schedule;
use crate::utils::pos;
use crate::value::Value;
use noisy_float::prelude::*;
use num::NumCast;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

pub trait Objective<'a, T, C, D>: Problem<T, C, D> + Sync
where
    T: Value<'a>,
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
}
impl<'a, T, C, D, P> Objective<'a, T, C, D> for P
where
    P: Problem<T, C, D> + Sync,
    T: Value<'a>,
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
}


