//! Norms.

use nalgebra::{DMatrix, DVector, RealField};
use nlopt::{Algorithm, Nlopt, Target};
use num::ToPrimitive;
use std::sync::Arc;

use crate::config::Config;
use crate::result::{Error, Result};
use crate::value::Value;
use crate::PRECISION;

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
        .ok_or(Error::MatrixMustBeInvertible)?;
    Ok(Arc::new(move |x: Config<T>| -> f64 {
        let d = DVector::from_vec((x - mean.clone()).to_vec());
        let result = d.transpose() * (&q_i * d);
        ToPrimitive::to_f64(&result[(0, 0)]).unwrap()
    }))
}

/// Computes the dual norm of `x` given some `norm`.
pub fn dual_norm(
    norm: &NormFn<'_, Config<f64>>,
    x: Config<f64>,
) -> Result<f64> {
    let d = x.d() as usize;

    let objective_function =
        |z: &[f64], _: Option<&mut [f64]>, _: &mut ()| -> f64 {
            Config::new(z.to_vec()) * x.clone()
        };
    let mut z = vec![0.; d];

    let mut opt = Nlopt::new(
        Algorithm::Bobyqa,
        d,
        objective_function,
        Target::Maximize,
        (),
    );
    opt.set_xtol_rel(PRECISION)?;

    opt.add_inequality_constraint(
        |z: &[f64], _: Option<&mut [f64]>, _: &mut ()| -> f64 {
            norm(Config::new(z.to_vec())) - 1.
        },
        (),
        PRECISION,
    )?;

    opt.optimize(&mut z)?;
    Ok(Config::new(z.to_vec()) * x.clone())
}
