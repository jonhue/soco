//! Norms.

use crate::config::{Config, FractionalConfig};
use crate::numerics::convex_optimization::{
    find_unbounded_maximizer, WrappedObjective,
};
use crate::result::{Failure, Result};
use crate::value::Value;
use crate::vec_wrapper::VecWrapper;
use nalgebra::{DMatrix, DVector, RealField};
use noisy_float::prelude::*;
use num::{NumCast, ToPrimitive};
use std::sync::Arc;

/// Norm function.
pub type NormFn<'a, T> = Arc<dyn Fn(Config<T>) -> N64 + Send + Sync + 'a>;

/// Manhattan norm.
pub fn manhattan<'a, T>() -> NormFn<'static, T>
where
    T: Value<'a>,
{
    Arc::new(|x: Config<T>| {
        n64(x
            .iter()
            .map(|j| ToPrimitive::to_f64(j).unwrap().abs())
            .sum())
    })
}

/// Manhattan norm scaled with switching costs.
pub fn manhattan_scaled<'a, T>(switching_cost: Vec<f64>) -> NormFn<'a, T>
where
    T: Value<'a>,
{
    Arc::new(move |x: Config<T>| {
        n64(x
            .iter()
            .enumerate()
            .map(|(k, j)| {
                switching_cost[k] / 2. * ToPrimitive::to_f64(j).unwrap().abs()
            })
            .sum())
    })
}

/// Euclidean norm.
pub fn euclidean<'a, T>() -> NormFn<'static, T>
where
    T: Value<'a>,
{
    Arc::new(|x: Config<T>| {
        n64(x
            .iter()
            .map(|j| ToPrimitive::to_f64(j).unwrap().powi(2))
            .sum::<f64>()
            .sqrt())
    })
}

/// Mahalanobis distance square. This norm is `1`-strongly convex and `1`-Lipschitz smooth.
///
/// For `Q` positive semi-definite.
pub fn mahalanobis<'a, T>(
    q: &DMatrix<T>,
    mean: Config<T>,
) -> Result<NormFn<'a, T>>
where
    T: RealField + Value<'a>,
{
    let q_i = q
        .clone()
        .try_inverse()
        .ok_or(Failure::MatrixMustBeInvertible)?;
    Ok(Arc::new(move |x: Config<T>| -> N64 {
        let d = DVector::from_vec((x - mean.clone()).to_vec());
        let result = d.transpose() * (&q_i * d);
        NumCast::from(result[(0, 0)]).unwrap()
    }))
}

/// Norm squared. `1`-strongly convex and `1`-Lipschitz smooth for the Euclidean norm and the Mahalanobis distance.
pub fn norm_squared<'a, T>(norm: &'a NormFn<'a, T>) -> NormFn<'a, T>
where
    T: Value<'a>,
{
    Arc::new(move |x: Config<T>| -> N64 { norm(x).powi(2) / n64(2.) })
}

/// Computes the dual norm of `x` given some `norm`.
pub fn dual<'a>(norm: &'a NormFn<'a, f64>) -> NormFn<'a, f64> {
    Arc::new(move |x: FractionalConfig| {
        let objective = WrappedObjective::new(x.clone(), |z, x| {
            n64(Config::new(z.to_vec()) * x.clone())
        });
        let constraint = WrappedObjective::new((), |z, _| {
            norm(Config::new(z.to_vec())) - n64(1.)
        });

        let (z, _) =
            find_unbounded_maximizer(objective, x.d(), vec![constraint]);
        n64(Config::new(z) * x)
    })
}
