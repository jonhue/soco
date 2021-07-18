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
