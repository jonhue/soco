use crate::problem::{Online, Problem};
use pyo3::{IntoPy, PyObject, Python};
use serde::{de::DeserializeOwned, Serialize};
use serde_derive::{Deserialize, Serialize};

pub mod data_center;

/// Model inputs to generate offline problem.
pub trait OfflineInput: std::fmt::Debug {}

/// Model inputs to update a problem instance (online) to the next time slot.
/// Encapsulates information for the current time slot as well as all time slots in the prediction window.
pub trait OnlineInput: std::fmt::Debug + DeserializeOwned + Serialize {}

/// Results of model.
pub trait ModelOutputSuccess:
    Clone + std::fmt::Debug + DeserializeOwned + IntoPy<PyObject> + Send + Serialize
{
    fn merge_with(self, output: Self) -> Self;
}
impl ModelOutputSuccess for () {
    fn merge_with(self, _: ()) {}
}
pub trait ModelOutputFailure:
    Clone + std::fmt::Debug + DeserializeOwned + IntoPy<PyObject> + Send + Serialize
{
    fn outside_decision_space() -> Self;
}
impl ModelOutputFailure for () {
    fn outside_decision_space() {}
}

#[derive(Clone, Debug, Deserialize, Serialize)]
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
    pub fn reduce(outputs: Vec<Self>) -> Self {
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
                        ModelOutput::Success(result.merge_with(output))
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
    assert!(span == o.w + 1, "There should be information for each time slot in the prediction window (`w = {}`) plus the current time slot. Got information for `{}` time slots.", o.w, span);
}
