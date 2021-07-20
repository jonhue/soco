//! Numerical computation.

use rayon::iter::{IntoParallelIterator, ParallelIterator};

pub mod convex_optimization;
pub mod finite_differences;
pub mod quadrature;
pub mod roots;

/// Precision used for numeric computations.
pub static PRECISION: f64 = 1e-6;
/// Tolerance used for numeric computations.
pub static TOLERANCE: f64 = PRECISION / 10.;

pub trait ApplicablePrecision {
    /// Rounds a value to precision.
    fn apply_precision(self) -> Self;
}
impl ApplicablePrecision for f64 {
    fn apply_precision(self) -> f64 {
        (self / PRECISION).round() * PRECISION
    }
}
impl<T> ApplicablePrecision for Vec<T>
where
    T: ApplicablePrecision + Send,
{
    fn apply_precision(self) -> Vec<T> {
        self.into_par_iter().map(T::apply_precision).collect()
    }
}
