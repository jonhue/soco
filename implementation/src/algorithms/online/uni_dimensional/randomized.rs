use crate::algorithms::online::{IntegralStep, OnlineAlgorithm, Step};
use crate::breakpoints::Breakpoints;
use crate::config::{Config, FractionalConfig};
use crate::convert::CastableSchedule;
use crate::model::{ModelOutputFailure, ModelOutputSuccess};
use crate::problem::{
    FractionalSimplifiedSmoothedConvexOptimization,
    IntegralSimplifiedSmoothedConvexOptimization, Online,
};
use crate::result::{Failure, Result};
use crate::schedule::IntegralSchedule;
use crate::utils::{assert, frac, project, sample_uniform};
use crate::{
    algorithms::online::{
        uni_dimensional::{
            probabilistic::{
                probabilistic, Memory as ProbabilisticMemory,
                Options as ProbabilisticOptions,
            },
            randomly_biased_greedy::{
                rbg, Memory as RandomlyBiasedGreedyMemory,
                Options as RandomlyBiasedGreedyOptions,
            },
        },
        FractionalStep,
    },
    schedule::FractionalSchedule,
};
use pyo3::prelude::*;
use serde_derive::{Deserialize, Serialize};
use std::marker::PhantomData;

/// Memory.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Memory<M> {
    /// Fractional number of servers determined by fractional relaxation.
    y: FractionalConfig,
    /// Memory of relaxation.
    relaxation_m: Option<M>,
}
impl<M> Default for Memory<M> {
    fn default() -> Self {
        Memory {
            y: Config::single(0.),
            relaxation_m: None,
        }
    }
}
impl<M> IntoPy<PyObject> for Memory<M>
where
    M: IntoPy<PyObject>,
{
    fn into_py(self, py: Python) -> PyObject {
        (self.y.to_vec(), self.relaxation_m).into_py(py)
    }
}

#[derive(Clone)]
pub struct Relaxation<M>(pub PhantomData<M>);
impl<'a> Default for Relaxation<ProbabilisticMemory<'a>> {
    fn default() -> Self {
        Self(PhantomData::<ProbabilisticMemory<'a>>)
    }
}
impl Default for Relaxation<RandomlyBiasedGreedyMemory> {
    fn default() -> Self {
        Self(PhantomData::<RandomlyBiasedGreedyMemory>)
    }
}

pub trait ExecutableRelaxation<'a, M, C, D> {
    fn execute(
        relaxation_o: Online<
            FractionalSimplifiedSmoothedConvexOptimization<'a, C, D>,
        >,
        xs: &FractionalSchedule,
        prev_m: Option<M>,
    ) -> Result<FractionalStep<M>>;
}
impl<'a, C, D> ExecutableRelaxation<'a, ProbabilisticMemory<'a>, C, D>
    for Relaxation<ProbabilisticMemory<'a>>
where
    C: ModelOutputSuccess + 'a,
    D: ModelOutputFailure + 'a,
{
    fn execute(
        relaxation_o: Online<
            FractionalSimplifiedSmoothedConvexOptimization<'a, C, D>,
        >,
        xs: &FractionalSchedule,
        prev_m: Option<ProbabilisticMemory<'a>>,
    ) -> Result<FractionalStep<ProbabilisticMemory<'a>>> {
        probabilistic.next(
            relaxation_o,
            xs,
            prev_m,
            ProbabilisticOptions {
                breakpoints: Breakpoints::grid(1.),
            },
        )
    }
}
impl<'a, C, D> ExecutableRelaxation<'a, RandomlyBiasedGreedyMemory, C, D>
    for Relaxation<RandomlyBiasedGreedyMemory>
where
    C: ModelOutputSuccess + 'a,
    D: ModelOutputFailure + 'a,
{
    fn execute(
        relaxation_o: Online<
            FractionalSimplifiedSmoothedConvexOptimization<'a, C, D>,
        >,
        xs: &FractionalSchedule,
        prev_m: Option<RandomlyBiasedGreedyMemory>,
    ) -> Result<FractionalStep<RandomlyBiasedGreedyMemory>> {
        rbg.next(
            relaxation_o.into_sco(),
            xs,
            prev_m,
            RandomlyBiasedGreedyOptions::default(),
        )
    }
}

/// Randomized Integral Relaxation
///
/// Relax discrete problem to fractional problem before use!
pub fn randomized<'a, M, C, D, R>(
    o: Online<IntegralSimplifiedSmoothedConvexOptimization<'a, C, D>>,
    _: i32,
    xs: &IntegralSchedule,
    prev_m: Memory<M>,
    _: R,
) -> Result<IntegralStep<Memory<M>>>
where
    C: ModelOutputSuccess + 'a,
    D: ModelOutputFailure + 'a,
    R: ExecutableRelaxation<'a, M, C, D>,
{
    assert(o.w == 0, Failure::UnsupportedPredictionWindow(o.w))?;
    assert(o.p.d == 1, Failure::UnsupportedProblemDimension(o.p.d))?;

    let relaxation_o = o.into_f();
    let Step(y, relaxation_m) =
        R::execute(relaxation_o, &xs.to(), prev_m.relaxation_m)?;

    let prev_x = xs.now_with_default(Config::single(0))[0];
    let prev_y = prev_m.y[0];

    let x = next(prev_x, prev_y, y[0]);
    let m = Memory { y, relaxation_m };

    Ok(Step(Config::single(x), Some(m)))
}

fn next(prev_x: i32, prev_y: f64, y: f64) -> i32 {
    #[allow(clippy::collapsible_else_if)]
    // Number of active servers increases (or remains the same).
    if prev_y <= y {
        if prev_x == y.ceil() as i32 {
            prev_x
        } else {
            let prev_y_proj = project(prev_y, y.floor(), y.ceil());
            let p = (y - prev_y_proj) / (1. - frac(prev_y_proj));

            let r = sample_uniform(0., 1.);
            if r <= p {
                y.ceil() as i32
            } else {
                y.floor() as i32
            }
        }
    }
    // Number of active servers decreases.
    else {
        if prev_x == y.floor() as i32 {
            prev_x
        } else {
            let prev_y_proj = project(prev_y, y.floor(), y.ceil());
            let p = (prev_y_proj - y) / frac(prev_y_proj);

            let r = sample_uniform(0., 1.);
            if r <= p {
                y.floor() as i32
            } else {
                y.ceil() as i32
            }
        }
    }
}
