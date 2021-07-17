//! Value trait.

use num::{Num, NumCast};
use serde::{Deserialize, Serialize};
use std::fmt::{Debug, Display};
use std::iter::Sum;

pub trait Value<'a>:
    Clone
    + Copy
    + Debug
    + Deserialize<'a>
    + Display
    + Num
    + NumCast
    + PartialOrd
    + Serialize
    + Sum
    + 'a
{
}

impl<'a, T> Value<'a> for T where
    T: Clone
        + Copy
        + Debug
        + Deserialize<'a>
        + Display
        + Num
        + NumCast
        + PartialOrd
        + Serialize
        + Sum
        + 'a
{
}
