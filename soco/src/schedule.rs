//! Definition of schedules.

use crate::config::Config;
use crate::utils::access;
use crate::value::Value;
use crate::vec_wrapper::VecWrapper;
use num::NumCast;
use serde_derive::{Deserialize, Serialize};
use std::iter::FromIterator;
use std::ops::Index;

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

    /// Builds a schedule from a raw (flat) encoding `raw_xs` (used for convex optimization).
    /// `d` is the number of dimensions, `w` is the length of the time window.
    /// The length of `raw_xs` must therefore be `d * (w + 1)`.
    pub fn from_raw(d: i32, w: i32, raw_xs: &[T]) -> Schedule<T> {
        assert_eq!(
            raw_xs.len() as i32,
            Schedule::<T>::raw_encoding_len(d, w),
            "length of raw encoding does not match expected length"
        );

        let mut xs = Schedule::empty();
        for t in 0..=w as usize {
            let i = d as usize * t;
            let x = Config::new(raw_xs[i..i + d as usize].to_vec());
            xs.push(x);
        }
        xs
    }

    /// Builds a raw (flat) encoding of a schedule (used for convex optimization) by stretching a config across the time window `w`.
    pub fn build_raw(w: i32, x: &Config<T>) -> Vec<T> {
        let l = Schedule::<T>::raw_encoding_len(x.d(), w) as usize;

        let mut raw_xs = vec![NumCast::from(0).unwrap(); l];
        for t in 0..=w as usize {
            let i = x.d() as usize * t;
            for k in 0..x.d() as usize {
                raw_xs[i + k] = x[k];
            }
        }
        raw_xs
    }

    /// Returns the length of the raw encoding of `d` dimensions across time window `w`.
    pub fn raw_encoding_len(d: i32, w: i32) -> i32 {
        d * (w + 1)
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
