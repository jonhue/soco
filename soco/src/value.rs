//! Value trait.

use num::{Num, NumCast};
use std::fmt::{Debug, Display};

pub trait Value:
    Clone + Copy + Debug + Display + Num + NumCast + PartialOrd
{
}

impl<T: Clone + Copy + Debug + Display + Num + NumCast + PartialOrd> Value
    for T
{
}
