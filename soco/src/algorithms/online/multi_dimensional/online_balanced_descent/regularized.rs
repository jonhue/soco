use crate::algorithms::online::{FractionalStep, Step};
use crate::config::{Config, FractionalConfig};
use crate::model::{ModelOutputFailure, ModelOutputSuccess};
use crate::numerics::convex_optimization::{
    find_minimizer, find_minimizer_of_hitting_cost, WrappedObjective,
};
use crate::problem::{FractionalSmoothedConvexOptimization, Online, Problem};
use crate::result::{Failure, Result};
use crate::schedule::FractionalSchedule;
use crate::utils::assert;

pub struct Options {
    /// Convexity parameter. Chosen such that `f_t(x) \geq f_t(v_t) + \frac{m}{2} \norm{x - v_t}_2^2` where `v_t` is the minimizer of `f_t`.
    pub m: f64,
    /// Convexity parameter of potential function of Bregman convergence.
    pub alpha: f64,
    /// Smoothness parameter of potential function of Bregman convergence.
    pub beta: f64,
}

#[derive(Clone)]
struct RegularizationFunctionObjectiveData<'a, C, D> {
    o: Online<FractionalSmoothedConvexOptimization<'a, C, D>>,
    t: i32,
    lambda_1: f64,
    lambda_2: f64,
    prev_x: FractionalConfig,
    v: FractionalConfig,
}

/// Regularized Online Balanced Descent
pub fn robd<C, D>(
    o: Online<FractionalSmoothedConvexOptimization<C, D>>,
    xs: &mut FractionalSchedule,
    _: &mut Vec<()>,
    options: &Options,
) -> Result<FractionalStep<()>>
where
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    assert(o.w == 0, Failure::UnsupportedPredictionWindow(o.w))?;

    let (lambda_1, lambda_2) =
        build_parameters(options.m, options.alpha, options.beta);

    let t = xs.t_end() + 1;
    let prev_x = if xs.is_empty() {
        Config::repeat(0., o.p.d)
    } else {
        xs.now()
    };

    let v = Config::new(
        find_minimizer_of_hitting_cost(
            t,
            o.p.hitting_cost.clone(),
            o.p.bounds.clone(),
        )
        .0,
    );
    let bounds = o.p.bounds.clone();
    let regularization_function = WrappedObjective::new(
        RegularizationFunctionObjectiveData {
            o,
            t,
            lambda_1,
            lambda_2,
            prev_x,
            v,
        },
        |x_, data| {
            let x = Config::new(x_.to_vec());
            data.o.p.hit_cost(data.t, x.clone()).cost
                + data.lambda_1
                    * (data.o.p.switching_cost)(x.clone() - data.prev_x.clone())
                        .raw()
                + data.lambda_2
                    * (data.o.p.switching_cost)(x - data.v.clone()).raw()
        },
    );
    let x = Config::new(find_minimizer(regularization_function, bounds).0);
    Ok(Step(x, None))
}

/// Determines `lambda_1` (weight of movement cost) and `lambda_2` (weight of regularizer).
fn build_parameters(m: f64, alpha: f64, beta: f64) -> (f64, f64) {
    let f_lambda_2 = |lambda_1| {
        (lambda_1 * m / 2.
            * (1. + (1. + 4. * beta.powi(2) / (alpha * m)).sqrt())
            - m)
            / beta
    };

    let mut lambda_2 = 0.;
    let mut lambda_1 =
        2. / (1. + (1. + 4. * beta.powi(2) / (alpha * m)).sqrt());
    if (f_lambda_2(lambda_1) - lambda_2).abs() < f64::EPSILON {
        return (lambda_1, lambda_2);
    }

    lambda_1 = 1.;
    lambda_2 = f_lambda_2(lambda_1);
    (lambda_1, lambda_2)
}
