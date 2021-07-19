use crate::problem::{Online, Problem};
use serde::{de::DeserializeOwned, Serialize};

pub mod data_center;

/// Model inputs to generate offline problem.
pub trait OfflineInput: std::fmt::Debug {}

/// Model inputs to update a problem instance (online) to the next time slot.
/// Encapsulates information for the current time slot as well as all time slots in the prediction window.
pub trait OnlineInput<'a>:
    std::fmt::Debug + DeserializeOwned + Serialize
{
}

/// Model which is used to generate problem instances and update them online.
pub trait Model<'a, P, A, B>: Clone
where
    P: Problem,
    A: OfflineInput,
    B: OnlineInput<'a>,
{
    /// Generates an offline problem instance given some `input` (with certainty).
    fn to(&'a self, input: A) -> P;

    /// Performs an online update of the given problem instance `o` with some `input` (which may be uncertain).
    fn update(&'a self, o: &mut Online<P>, input: B);
}

/// Utility to verify that the update of an online instance is valid.
pub fn verify_update<P>(o: &Online<P>, span: i32)
where
    P: Problem,
{
    assert!(span == o.w + 1, "There should be information for each time slot in the prediction window (`w = {}`) plus the current time slot. Got information for `{}` time slots.", o.w, span);
}
