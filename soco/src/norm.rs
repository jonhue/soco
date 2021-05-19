use num::ToPrimitive;
use std::sync::Arc;

use crate::config::Config;
use crate::value::Value;

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
