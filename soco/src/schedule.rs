//! Definition of schedules.

use std::iter::FromIterator;
use std::ops::Index;

use crate::config::Config;
use crate::utils::access;
use crate::vec_wrapper::VecWrapper;

/// Includes all configurations from time `1` to time `t_end`.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Schedule<T>(pub Vec<Config<T>>);
pub type IntegralSchedule = Schedule<i32>;
pub type FractionalSchedule = Schedule<f64>;

impl<T> Schedule<T>
where
    T: Clone,
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
}

impl<T> Index<usize> for Schedule<T> {
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

impl<T> VecWrapper for Schedule<T> {
    type Item = Config<T>;

    fn to_vec(&self) -> &Vec<Self::Item> {
        &self.0
    }
}

impl<T> FromIterator<Config<T>> for Schedule<T>
where
    T: Clone,
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
