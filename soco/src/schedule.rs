//! Definition of schedules.

use crate::utils::access;
use crate::value::Value;
use crate::{config::Config, vec_wrapper::VecWrapper};
use rayon::{
    iter::{
        FromParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    },
    slice::Iter,
    vec::IntoIter,
};
use serde_derive::{Deserialize, Serialize};
use std::{iter::FromIterator, ops::Index};

/// Includes all configurations from time `1` to time `t_end`.
#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct Schedule<T>(Vec<Config<T>>);
pub type IntegralSchedule = Schedule<i32>;
pub type FractionalSchedule = Schedule<f64>;

impl<'a, T> Schedule<T>
where
    T: Value<'a>,
{
    pub fn new(x: Vec<Config<T>>) -> Schedule<T> {
        Schedule(x)
    }

    pub fn empty() -> Schedule<T> {
        Schedule(vec![])
    }

    /// Returns `true` if the schedule is empty (i.e. does not include any config).
    pub fn is_empty(&self) -> bool {
        self.t_end() == 0
    }

    /// Returns the time of the latest time step.
    pub fn t_end(&self) -> i32 {
        self.0.len() as i32
    }

    /// Returns the config of the latest time step.
    pub fn now(&self) -> Config<T> {
        self[self.0.len() - 1].clone()
    }

    /// Returns the config of the latest time step.
    pub fn now_with_default(&self, default: Config<T>) -> Config<T> {
        if self.is_empty() {
            default
        } else {
            self[self.0.len() - 1].clone()
        }
    }

    /// Returns the config at time `t` if present.
    pub fn get(&self, t: i32) -> Option<&Config<T>> {
        access(&self.0, t)
    }

    /// Extends schedule with a new initial config.
    pub fn shift(&mut self, x: Config<T>) {
        self.0.insert(0, x)
    }

    /// Extends schedule with a new final config.
    pub fn push(&mut self, x: Config<T>) {
        self.0.push(x)
    }

    /// Immutably Extends schedule with a new final config.
    pub fn extend(&self, x: Config<T>) -> Schedule<T> {
        Schedule([&self.0[..], &[x]].concat())
    }

    /// Converts schedule to a vector of vectors.
    pub fn to_vec(&self) -> Vec<Vec<T>> {
        self.0.iter().map(|x| x.to_vec()).collect()
    }

    /// Builds a schedule from a raw (flat) encoding `raw_xs` (used for convex optimization).
    /// `d` is the number of dimensions, `w` is the length of the time window.
    /// The length of `raw_xs` must therefore be `d * w`.
    pub fn from_raw(d: i32, w: i32, raw_xs: &[T]) -> Schedule<T> {
        assert_eq!(
            raw_xs.len() as i32,
            Schedule::<T>::raw_encoding_len(d, w),
            "length of raw encoding does not match expected length"
        );

        Schedule::new(
            (0..w as usize)
                .into_iter()
                .map(|t| {
                    let i = d as usize * t;
                    Config::new(raw_xs[i..i + d as usize].to_vec())
                })
                .collect(),
        )
    }

    /// Builds a raw (flat) encoding of a schedule (used for convex optimization) by stretching a config across the time window `w`.
    pub fn build_raw(w: i32, x: &Config<T>) -> Vec<T> {
        let raw_xs: Vec<T> = (0..w as usize)
            .into_iter()
            .flat_map(|_| x.iter().cloned())
            .collect();
        assert_eq!(
            raw_xs.len() as i32,
            Schedule::<T>::raw_encoding_len(x.d(), w),
            "length of raw encoding does not match expected length"
        );
        raw_xs
    }

    /// Returns the length of the raw encoding of `d` dimensions across time window `w`.
    pub fn raw_encoding_len(d: i32, w: i32) -> i32 {
        d * w
    }
}

impl<'a, T> Index<usize> for Schedule<T>
where
    T: Value<'a>,
{
    type Output = Config<T>;

    fn index(&self, t: usize) -> &Config<T> {
        assert!(
            t < self.0.len(),
            "argument must denote one of {} time steps, is {}",
            self.0.len(),
            t + 1
        );
        &self.0[t]
    }
}

impl<'a, T> VecWrapper for Schedule<T>
where
    T: Value<'a>,
{
    type Item = Config<T>;

    fn to_vec(&self) -> &Vec<Self::Item> {
        &self.0
    }
}

impl<'a, T> FromIterator<Config<T>> for Schedule<T>
where
    T: Value<'a>,
{
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = Config<T>>,
    {
        let mut xs = vec![];
        for x in iter {
            xs.push(x);
        }
        Schedule::new(xs)
    }
}

impl<'a, T> FromParallelIterator<Config<T>> for Schedule<T>
where
    T: Value<'a>,
{
    fn from_par_iter<I>(iter: I) -> Self
    where
        I: IntoParallelIterator<Item = Config<T>>,
    {
        Schedule::new(Vec::<Config<T>>::from_par_iter(iter))
    }
}

impl<'a, 'b, T> IntoParallelIterator for &'a Schedule<T>
where
    T: Value<'b>,
{
    type Item = &'a Config<T>;
    type Iter = Iter<'a, Config<T>>;

    fn into_par_iter(self) -> Self::Iter {
        self.0.par_iter()
    }
}

impl<'a, T> IntoParallelIterator for Schedule<T>
where
    T: Value<'a>,
{
    type Item = Config<T>;
    type Iter = IntoIter<Config<T>>;

    fn into_par_iter(self) -> Self::Iter {
        self.0.into_par_iter()
    }
}
