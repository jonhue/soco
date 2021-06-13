//! Definition of schedules.

use crate::config::Config;
use crate::utils::access;
use crate::value::Value;
use crate::vec_wrapper::VecWrapper;
use num::NumCast;
use std::iter::FromIterator;
use std::ops::Index;

/// Includes all configurations from time `1` to time `t_end`.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Schedule<T>(pub Vec<Config<T>>)
where
    T: Value;
pub type IntegralSchedule = Schedule<i32>;
pub type FractionalSchedule = Schedule<f64>;

impl<T> Schedule<T>
where
    T: Value,
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
    pub fn now(&self) -> &Config<T> {
        &self[self.0.len() - 1]
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
            d * (w + 1),
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
        let l = (x.d() * (w + 1)) as usize;

        let mut raw_xs = vec![NumCast::from(0).unwrap(); l];
        for t in 0..=w as usize {
            let i = x.d() as usize * t;
            for k in 0..x.d() as usize {
                raw_xs[i + k] = x[k];
            }
        }
        raw_xs
    }
}

impl<T> Index<usize> for Schedule<T>
where
    T: Value,
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

impl<T> VecWrapper for Schedule<T>
where
    T: Value,
{
    type Item = Config<T>;

    fn to_vec(&self) -> &Vec<Self::Item> {
        &self.0
    }
}

impl<T> FromIterator<Config<T>> for Schedule<T>
where
    T: Value,
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
