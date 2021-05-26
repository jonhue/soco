//! Value trait.

use num::{Num, NumCast};
use std::fmt::{Debug, Display};
use std::iter::Sum;

pub trait Value:
    Clone + Copy + Debug + Display + Num + NumCast + PartialOrd + Sum
{
}

impl<T: Clone + Copy + Debug + Display + Num + NumCast + PartialOrd + Sum> Value
    for T
{
}
