//! Norms.

use crate::config::{Config, FractionalConfig};
use crate::numerics::convex_optimization::find_unbounded_maximizer;
use crate::result::{Failure, Result};
use crate::value::Value;
use nalgebra::{DMatrix, DVector, RealField};
use num::ToPrimitive;
use std::sync::Arc;

/// Norm function.
pub type NormFn<'a, T> = Arc<dyn Fn(T) -> f64 + 'a>;

/// Manhattan norm.
pub fn manhattan<T>(x: Config<T>) -> f64
where
    T: Value,
{
    let mut result = 0.;
    for k in 0..x.d() as usize {
        result += ToPrimitive::to_f64(&x[k]).unwrap().abs();
    }
    result
}

/// Manhattan norm scaled with switching costs.
pub fn manhattan_scaled<T>(x: Config<T>, switching_cost: Vec<f64>) -> f64
where
    T: Value,
{
    let mut result = 0.;
    for k in 0..x.d() as usize {
        result +=
            switching_cost[k] / 2. + ToPrimitive::to_f64(&x[k]).unwrap().abs();
    }
    result
}

/// Euclidean norm.
pub fn euclidean<T>(x: Config<T>) -> f64
where
    T: Value,
{
    let mut result = 0.;
    for k in 0..x.d() as usize {
        result += ToPrimitive::to_f64(&x[k]).unwrap().powi(2);
    }
    result.sqrt()
}

/// Mahalanobis distance square. This norm is `1`-strongly convex and `1`-Lipschitz smooth.
///
/// For `Q` positive semi-definite.
pub fn mahalanobis<'a, T>(
    q: &DMatrix<T>,
    mean: Config<T>,
) -> Result<NormFn<'a, Config<T>>>
where
    T: RealField + Value,
{
    let q_i = q
        .clone()
        .try_inverse()
        .ok_or(Failure::MatrixMustBeInvertible)?;
    Ok(Arc::new(move |x: Config<T>| -> f64 {
        let d = DVector::from_vec((x - mean.clone()).to_vec());
        let result = d.transpose() * (&q_i * d);
        ToPrimitive::to_f64(&result[(0, 0)]).unwrap()
    }))
}

/// Norm squared. `1`-strongly convex and `1`-Lipschitz smooth for the Euclidean norm and the Mahalanobis distance.
pub fn norm_squared<'a, T>(norm: NormFn<'a, Config<T>>) -> NormFn<'a, Config<T>>
where
    T: Value + 'a,
{
    Arc::new(move |x: Config<T>| -> f64 { norm(x).powi(2) / 2. })
}

/// Computes the dual norm of `x` given some `norm`.
pub fn dual(
    norm: &NormFn<'_, FractionalConfig>,
    x: FractionalConfig,
) -> Result<f64> {
    let objective = |z: &[f64]| -> f64 { Config::new(z.to_vec()) * x.clone() };
    let constraint =
        Arc::new(|z: &[f64]| -> f64 { norm(Config::new(z.to_vec())) - 1. });

    let (z, _) =
        find_unbounded_maximizer(objective, x.d(), vec![constraint], vec![])?;
    Ok(Config::new(z) * x)
}
