//! Utilities to model cost of data centers.

use std::{iter::Sum, ops::Add};

use super::{ModelOutput, ModelOutputFailure, ModelOutputSuccess};
use noisy_float::prelude::*;
use pyo3::prelude::*;
use serde_derive::{Deserialize, Serialize};
use thiserror::Error;

pub mod loads;
pub mod model;
pub mod models;

#[pyclass]
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct DataCenterModelOutputSuccess {
    /// Energy cost of model.
    #[pyo3(get)]
    pub energy_cost: f64,
    /// Revenue loss of model.
    #[pyo3(get)]
    pub revenue_loss: f64,
    /// All possible assignments of fractions of loads to server types for each time slot.
    #[pyo3(get)]
    pub assignments: Vec<Vec<Vec<f64>>>,
}
impl ModelOutputSuccess for DataCenterModelOutputSuccess {
    fn horizontal_merge(mut self, output: Self) -> Self {
        self.assignments.extend(output.assignments.into_iter());
        Self {
            energy_cost: self.energy_cost + output.energy_cost,
            revenue_loss: self.revenue_loss + output.revenue_loss,
            assignments: self.assignments,
        }
    }

    /// Computes the mean energy cost and revenue loss.
    fn vertical_merge(mut self, output: Self) -> Self {
        assert!(self.assignments.len() == 1);
        assert!(output.assignments.len() == 1);
        self.assignments[0]
            .extend(output.assignments.into_iter().next().unwrap().into_iter());
        Self {
            energy_cost: (self.energy_cost + output.energy_cost) / 2.,
            revenue_loss: (self.revenue_loss + output.revenue_loss) / 2.,
            assignments: self.assignments,
        }
    }
}
impl DataCenterModelOutputSuccess {
    pub fn new(
        energy_cost: f64,
        revenue_loss: f64,
        assignment: Vec<f64>,
    ) -> Self {
        Self {
            energy_cost,
            revenue_loss,
            assignments: vec![vec![assignment]],
        }
    }
}

#[derive(Clone, Debug, Error, Deserialize, PartialEq, Serialize)]
pub enum DataCenterModelOutputFailure {
    #[error("The configuration is unable to support the given load profile.")]
    DemandExceedingSupply,
    #[error("The delay is infinite. The arrival rate is too close to (or larger than) the service rate.")]
    InfiniteDelay {
        server_type: String,
        number_of_jobs: f64,
        mean_job_duration: f64,
    },
    #[error("A positive load was assigned to a server type without any active servers.")]
    LoadToInactiveServer,
    #[error("The configuration is outside the decision space.")]
    OutsideDecisionSpace,
    #[error("A server cannot handle more than one job during a time slot.")]
    SLOMaxUtilizationExceeded,
}
impl IntoPy<PyObject> for DataCenterModelOutputFailure {
    fn into_py(self, py: Python) -> PyObject {
        self.to_string().into_py(py)
    }
}
impl ModelOutputFailure for DataCenterModelOutputFailure {
    fn outside_decision_space() -> DataCenterModelOutputFailure {
        DataCenterModelOutputFailure::OutsideDecisionSpace
    }
}

pub type DataCenterModelOutput =
    ModelOutput<DataCenterModelOutputSuccess, DataCenterModelOutputFailure>;

#[derive(Debug)]
pub struct DataCenterObjective {
    energy_cost: N64,
    revenue_loss: N64,
}
impl DataCenterObjective {
    pub fn new(energy_cost: N64, revenue_loss: N64) -> Self {
        Self {
            energy_cost,
            revenue_loss,
        }
    }

    pub fn failure(_failure: DataCenterModelOutputFailure) -> Self {
        Self {
            energy_cost: n64(f64::INFINITY),
            revenue_loss: n64(f64::INFINITY),
        }
    }
}
impl Default for DataCenterObjective {
    fn default() -> Self {
        Self {
            energy_cost: n64(0.),
            revenue_loss: n64(0.),
        }
    }
}
impl Add for DataCenterObjective {
    type Output = DataCenterObjective;
    fn add(self, rhs: Self) -> Self::Output {
        DataCenterObjective {
            energy_cost: self.energy_cost + rhs.energy_cost,
            revenue_loss: self.revenue_loss + rhs.revenue_loss,
        }
    }
}
impl Sum for DataCenterObjective {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|result, value| result + value)
            .unwrap_or_default()
    }
}

pub type IntermediateResult = Result<N64, DataCenterModelOutputFailure>;

pub type IntermediateObjective =
    Result<DataCenterObjective, DataCenterModelOutputFailure>;

/// Ensures that `x` is greater than zero and handles edge cases appropriately.
pub fn safe_balancing(
    x: N64,
    total_load: N64,
    f: impl Fn() -> IntermediateResult,
) -> IntermediateResult {
    if x > 0. {
        f()
    } else if total_load > 0. {
        Err(DataCenterModelOutputFailure::LoadToInactiveServer)
    } else {
        assert!(total_load == 0.);
        Ok(n64(0.))
    }
}
