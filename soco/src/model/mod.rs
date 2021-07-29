use std::panic::UnwindSafe;

use crate::problem::{Online, Problem};
use log::{info};
use pyo3::{IntoPy, PyObject, Python};
use serde::{de::DeserializeOwned, Serialize};
use serde_derive::{Deserialize, Serialize};

pub mod data_center;

/// Model inputs to generate offline problem.
pub trait OfflineInput: std::fmt::Debug {}

/// Model inputs to update a problem instance (online) to the next time slot.
/// Encapsulates information for the current time slot as well as all time slots in the prediction window.
pub trait OnlineInput:
    std::fmt::Debug + DeserializeOwned + Serialize + UnwindSafe
{
}

/// Results of model.
pub trait ModelOutputSuccess:
    Clone
    + std::fmt::Debug
    + DeserializeOwned
    + IntoPy<PyObject>
    + Send
    + Serialize
{
    /// Merge two outputs across time steps.
    fn horizontal_merge(self, output: Self) -> Self;

    /// Merge two outputs within the same time step.
    fn vertical_merge(self, output: Self) -> Self;
}
impl ModelOutputSuccess for () {
    fn horizontal_merge(self, _: ()) {}
    fn vertical_merge(self, _: ()) {}
}
pub trait ModelOutputFailure:
    Clone
    + std::fmt::Debug
    + DeserializeOwned
    + IntoPy<PyObject>
    + Send
    + Serialize
{
    fn outside_decision_space() -> Self;
}
impl ModelOutputFailure for () {
    fn outside_decision_space() {}
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub enum ModelOutput<C, D> {
    Success(C),
    Failure(D),
    None,
}
impl<C, D> IntoPy<PyObject> for ModelOutput<C, D>
where
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    fn into_py(self, py: Python) -> PyObject {
        match self {
            Self::Success(value) => value.into_py(py),
            Self::Failure(value) => value.into_py(py),
            Self::None => ().into_py(py),
        }
    }
}
impl<C, D> ModelOutput<C, D>
where
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    pub fn horizontal_reduce(outputs: Vec<Self>) -> Self {
        assert!(!outputs.is_empty());
        outputs
            .into_iter()
            .reduce(|result, output| match result {
                ModelOutput::Failure(result) => ModelOutput::Failure(result),
                ModelOutput::Success(result) => match output {
                    ModelOutput::Failure(output) => {
                        ModelOutput::Failure(output)
                    }
                    ModelOutput::Success(output) => {
                        ModelOutput::Success(result.horizontal_merge(output))
                    }
                    ModelOutput::None => ModelOutput::Success(result),
                },
                ModelOutput::None => output,
            })
            .unwrap()
    }

    pub fn vertical_reduce(outputs: Vec<Self>) -> Self {
        assert!(!outputs.is_empty());
        outputs
            .into_iter()
            .reduce(|result, output| match result {
                ModelOutput::Failure(result) => ModelOutput::Failure(result),
                ModelOutput::Success(result) => match output {
                    ModelOutput::Failure(output) => {
                        ModelOutput::Failure(output)
                    }
                    ModelOutput::Success(output) => {
                        ModelOutput::Success(result.vertical_merge(output))
                    }
                    ModelOutput::None => ModelOutput::Success(result),
                },
                ModelOutput::None => output,
            })
            .unwrap()
    }
}

/// Model which is used to generate problem instances and update them online.
pub trait Model<T, P, A, B, C, D>: Clone + Send + Sync
where
    P: Problem<T, C, D>,
    A: OfflineInput,
    B: OnlineInput,
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    /// Generates an offline problem instance given some `input` (with certainty).
    fn to(&self, input: A) -> P;

    /// Performs an online update of the given problem instance `o` with some `input` (which may be uncertain).
    fn update(&self, o: &mut Online<P>, input: B);
}

/// Utility to verify that the update of an online instance is valid.
pub fn verify_update<T, P, C, D>(o: &Online<P>, span: i32)
where
    P: Problem<T, C, D>,
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    assert!(span >= o.w + 1, "There should be information for each time slot in the prediction window (`w = {}`) plus the current time slot. Got information for `{}` time slots.", o.w, span);
    if span > o.w + 1 {
        info!("The inputs have prediction window `{}` which is not used completely by the algorithm with prediction window `{}`. Consider using a different algorithm.", span, o.w);
    }
}
