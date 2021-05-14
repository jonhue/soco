//! Definition of configurations.

use std::iter::FromIterator;
use std::ops::Index;

use crate::vec_wrapper::VecWrapper;

/// For some time `t`, assigns each dimension `d` a unique value.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Config<T>(Vec<T>);

impl<T> Config<T>
where
    T: Clone,
{
    pub fn new(x: Vec<T>) -> Config<T> {
        Config(x)
    }

    pub fn single(j: T) -> Config<T> {
        Config(vec![j])
    }

    pub fn repeat(j: T, d: i32) -> Config<T>
    where
        T: Clone,
    {
        Config(vec![j; d as usize])
    }

    pub fn to_vec(&self) -> Vec<T> {
        self.0.clone()
    }
}

impl<T> Index<usize> for Config<T> {
    type Output = T;

    fn index(&self, k: usize) -> &T {
        assert!(
            k < self.0.len(),
            "argument must denote one of {} dimensions, is {}",
            self.0.len(),
            k + 1
        );
        &self.0[k]
    }
}

impl<T> VecWrapper for Config<T> {
    type Item = T;

    fn to_vec(&self) -> &Vec<Self::Item> {
        &self.0
    }
}

impl<T> FromIterator<T> for Config<T>
where
    T: Clone,
{
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let mut x = vec![];
        for j in iter {
            x.push(j);
        }
        Config::new(x)
    }
}