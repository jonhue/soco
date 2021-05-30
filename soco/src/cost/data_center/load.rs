//! Definition of loads.

use std::iter::FromIterator;
use std::ops::Div;
use std::ops::Index;
use std::ops::Mul;

use crate::vec_wrapper::VecWrapper;

/// For some time `t`, encapsulates the load of `e` types.
#[derive(Clone, Debug, PartialEq)]
pub struct Load(Vec<f64>);

impl Load {
    pub fn new(l: Vec<f64>) -> Load {
        Load(l)
    }

    pub fn single(l: f64) -> Load {
        Load(vec![l])
    }

    pub fn e(&self) -> i32 {
        self.0.len() as i32
    }

    pub fn total(&self) -> f64 {
        self.0.iter().copied().sum::<f64>()
    }

    pub fn to_vec(&self) -> Vec<f64> {
        self.0.clone()
    }
}

impl Index<usize> for Load {
    type Output = f64;

    fn index(&self, i: usize) -> &f64 {
        assert!(
            i < self.0.len(),
            "argument must denote one of {} types, is {}",
            self.0.len(),
            i + 1
        );
        &self.0[i]
    }
}

impl VecWrapper for Load {
    type Item = f64;

    fn to_vec(&self) -> &Vec<Self::Item> {
        &self.0
    }
}

impl FromIterator<f64> for Load {
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = f64>,
    {
        let mut l = vec![];
        for j in iter {
            l.push(j);
        }
        Load::new(l)
    }
}

impl Mul<Vec<f64>> for Load {
    type Output = Load;

    /// Applies fractions to loads.
    fn mul(self, z: Vec<f64>) -> Self::Output {
        self.iter().zip(&z).map(|(&l, &z)| l * z).collect::<Load>()
    }
}

impl Div<f64> for Load {
    type Output = Load;

    /// Divides loads by scalar.
    fn div(self, other: f64) -> Self::Output {
        self.iter().map(|&l| l / other).collect()
    }
}
