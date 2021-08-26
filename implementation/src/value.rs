//! Value trait.

use num::{Num, NumCast};
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::fmt::{Debug, Display};
use std::iter::Sum;

pub trait Value<'a>:
    Clone
    + Copy
    + Debug
    + DeserializeOwned
    + Discretizable
    + Display
    + Num
    + NumCast
    + PartialOrd
    + Send
    + Serialize
    + Sum
    + Sync
    + 'a
{
}

impl<'a, T> Value<'a> for T where
    T: Clone
        + Copy
        + Debug
        + DeserializeOwned
        + Discretizable
        + Display
        + Num
        + NumCast
        + PartialOrd
        + Send
        + Serialize
        + Sum
        + Sync
        + 'a
{
}

pub trait Discretizable {
    fn ceil(self) -> i32;
    fn floor(self) -> i32;
}

impl Discretizable for i32 {
    fn ceil(self) -> i32 {
        self
    }

    fn floor(self) -> i32 {
        self
    }
}

impl Discretizable for f64 {
    fn ceil(self) -> i32 {
        self.ceil() as i32
    }

    fn floor(self) -> i32 {
        self.floor() as i32
    }
}

impl Discretizable for usize {
    fn ceil(self) -> i32 {
        self as i32
    }

    fn floor(self) -> i32 {
        self as i32
    }
}
