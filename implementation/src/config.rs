//! Definition of configurations (points in the decision space).

use crate::value::Value;
use crate::vec_wrapper::VecWrapper;
use pyo3::prelude::*;
use rayon::iter::{
    FromParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
};
use rayon::slice::Iter;
use rayon::vec::IntoIter;
use serde_derive::{Deserialize, Serialize};
use std::iter::FromIterator;
use std::ops::Add;
use std::ops::Div;
use std::ops::Index;
use std::ops::IndexMut;
use std::ops::Mul;
use std::ops::Sub;

/// Assigns each dimension $k \in \[d\]$ a unique value.
#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct Config<T>(Vec<T>);
pub type IntegralConfig = Config<i32>;
pub type FractionalConfig = Config<f64>;

impl<'a, T> Config<T>
where
    T: Value<'a>,
{
    /// Converts a vector to a configuration.
    pub fn new(x: Vec<T>) -> Config<T> {
        Config(x)
    }

    /// Creates an empty configuration.
    pub fn empty() -> Config<T> {
        Config(vec![])
    }

    /// Creates a uni-dimensional configuration.
    pub fn single(j: T) -> Config<T> {
        Config(vec![j])
    }

    /// Creates a configuration by repeating `j` across `d` dimensions.
    pub fn repeat(j: T, d: i32) -> Config<T>
    where
        T: Clone,
    {
        Config(vec![j; d as usize])
    }

    /// Returns the dimensionality of a configuration.
    pub fn d(&self) -> i32 {
        self.0.len() as i32
    }

    /// Clones the inner vector.
    pub fn to_vec(&self) -> Vec<T> {
        self.0.clone()
    }

    /// Appends a value `j` to the configuration.
    pub fn push(&mut self, j: T) {
        self.0.push(j)
    }

    /// Returns the sum of values across all dimensions.
    pub fn total(&self) -> T {
        self.0.clone().into_iter().sum()
    }
}

impl<'a, T> FromPyObject<'a> for Config<T>
where
    T: Value<'a> + FromPyObject<'a>,
{
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        Ok(Config::new(ob.extract()?))
    }
}

impl<'a, T> IntoPy<PyObject> for Config<T>
where
    T: Value<'a> + IntoPy<PyObject>,
{
    fn into_py(self, py: Python) -> PyObject {
        self.to_vec().into_py(py)
    }
}

impl<'a, T> Index<usize> for Config<T>
where
    T: Value<'a>,
{
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

impl<'a, T> IndexMut<usize> for Config<T>
where
    T: Value<'a>,
{
    fn index_mut(&mut self, k: usize) -> &mut T {
        assert!(
            k < self.0.len(),
            "argument must denote one of {} dimensions, is {}",
            self.0.len(),
            k + 1
        );
        &mut self.0[k]
    }
}

impl<'a, T> VecWrapper for Config<T>
where
    T: Value<'a>,
{
    type Item = T;

    fn to_vec(&self) -> &Vec<Self::Item> {
        &self.0
    }
}

impl<'a, T> FromIterator<T> for Config<T>
where
    T: Value<'a>,
{
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        Config::new(Vec::<T>::from_iter(iter))
    }
}

impl<'a, T> FromParallelIterator<T> for Config<T>
where
    T: Value<'a>,
{
    fn from_par_iter<I>(iter: I) -> Self
    where
        I: IntoParallelIterator<Item = T>,
    {
        Config::new(Vec::<T>::from_par_iter(iter))
    }
}

impl<'data, T> IntoParallelIterator for &'data Config<T>
where
    T: Value<'data>,
{
    type Item = &'data T;
    type Iter = Iter<'data, T>;

    fn into_par_iter(self) -> Self::Iter {
        self.0.par_iter()
    }
}

impl<'a, T> IntoParallelIterator for Config<T>
where
    T: Value<'a>,
{
    type Item = T;
    type Iter = IntoIter<T>;

    fn into_par_iter(self) -> Self::Iter {
        self.0.into_par_iter()
    }
}

impl<'a, T> Add for Config<T>
where
    T: Value<'a>,
{
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        self.iter()
            .zip(other.iter())
            .map(|(&x, &y)| x + y)
            .collect()
    }
}

impl<'a, T> Sub for Config<T>
where
    T: Value<'a>,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        self.iter()
            .zip(other.iter())
            .map(|(&x, &y)| x - y)
            .collect()
    }
}

impl<'a, T> Mul for Config<T>
where
    T: Value<'a>,
{
    type Output = T;

    /// Dot product of transposed `self` with `other`.
    fn mul(self, other: Self) -> Self::Output {
        self.iter().zip(other.iter()).map(|(&x, &y)| x * y).sum()
    }
}

impl<'a> Mul<FractionalConfig> for f64 {
    type Output = FractionalConfig;

    /// Scales config with scalar.
    fn mul(self, other: FractionalConfig) -> Self::Output {
        other.iter().map(|&j| self * j).collect()
    }
}

impl<'a> Div<f64> for FractionalConfig {
    type Output = FractionalConfig;

    /// Divides config by scalar.
    fn div(self, other: f64) -> Self::Output {
        self.iter().map(|&j| j / other).collect()
    }
}
