//! Switching cost model.

use crate::model::data_center::model::ServerType;
use noisy_float::prelude::*;
use pyo3::prelude::*;
use std::collections::HashMap;

/// Switching cost model. Parameters are provided separately for each server type.
#[derive(Clone, FromPyObject)]
pub struct SwitchingCostModel(HashMap<String, SwitchingCost>);

/// Switching cost.
#[pyclass]
#[derive(Clone)]
pub struct SwitchingCost {
    /// Average cost per unit of energy.
    #[pyo3(get, set)]
    pub energy_cost: f64,
    /// Power consumed when idling.
    #[pyo3(get, set)]
    pub phi_min: f64,
    /// Power consumed at full load.
    #[pyo3(get, set)]
    pub phi_max: f64,
    /// Additional energy consumed by toggling a server on and off.
    #[pyo3(get, set)]
    pub epsilon: f64,
    /// Required time in time slots for migrating connections or data.
    #[pyo3(get, set)]
    pub delta: f64,
    /// Wear-and-tear costs of toggling a server.
    #[pyo3(get, set)]
    pub tau: f64,
    /// Perceived risk associated with toggling a server.
    #[pyo3(get, set)]
    pub rho: f64,
}
impl SwitchingCost {
    /// Computes switching cost for a server of some type.
    pub fn switching_cost(&self) -> N64 {
        n64(
            self.energy_cost * (self.epsilon + self.delta * self.phi_max)
                + self.tau
                + self.rho,
        )
    }

    /// Computes normalized switching cost for a server of some type given the time slot length $\delta$.
    /// Approximately, measures the minimum duration a server must be asleep to outweigh the switching cost.
    /// Referred to as $\xi$ in the paper.
    pub fn normalized_switching_cost(&self, delta: f64) -> N64 {
        self.switching_cost() / (self.energy_cost * delta * self.phi_min)
    }

    /// Builds switching cost such that the normalized switching cost matches $normalized_switching_cost$.
    /// Here, $\delta$ is the time slot length.
    pub fn from_normalized(
        delta: f64,
        normalized_switching_cost: f64,
        energy_cost: f64,
        phi_min: f64,
    ) -> Self {
        Self {
            energy_cost,
            phi_min,
            phi_max: 0.,
            epsilon: 0.,
            delta: 0.,
            tau: 0.,
            rho: normalized_switching_cost * energy_cost * delta * phi_min,
        }
    }
}
#[pymethods]
impl SwitchingCost {
    #[new]
    fn constructor(
        energy_cost: f64,
        phi_min: f64,
        phi_max: f64,
        epsilon: f64,
        delta: f64,
        tau: f64,
        rho: f64,
    ) -> Self {
        SwitchingCost {
            energy_cost,
            phi_min,
            phi_max,
            epsilon,
            delta,
            tau,
            rho,
        }
    }

    #[pyo3(name = "switching_cost")]
    fn switching_cost_py(&self) -> PyResult<f64> {
        Ok(self.switching_cost().raw())
    }

    #[pyo3(name = "normalized_switching_cost")]
    fn normalized_switching_cost_py(&self, delta: f64) -> PyResult<f64> {
        Ok(self.normalized_switching_cost(delta).raw())
    }

    #[staticmethod]
    #[pyo3(name = "from_normalized")]
    fn from_normalized_py(
        delta: f64,
        normalized_switching_cost: f64,
        energy_cost: f64,
        phi_min: f64,
    ) -> PyResult<Self> {
        Ok(Self::from_normalized(
            delta,
            normalized_switching_cost,
            energy_cost,
            phi_min,
        ))
    }
}

impl SwitchingCostModel {
    pub fn new(switching_costs: HashMap<String, SwitchingCost>) -> Self {
        SwitchingCostModel(switching_costs)
    }

    /// Builds vector of switching costs for all server types.
    pub fn switching_costs(&self, server_types: &Vec<ServerType>) -> Vec<f64> {
        server_types
            .iter()
            .map(|server_type| self.model(server_type).switching_cost().raw())
            .collect()
    }

    /// Builds vector of normalized switching costs for all server types.
    pub fn normalized_switching_costs(
        &self,
        delta: f64,
        server_types: &Vec<ServerType>,
    ) -> Vec<f64> {
        server_types
            .iter()
            .map(|server_type| {
                self.model(server_type)
                    .normalized_switching_cost(delta)
                    .raw()
            })
            .collect()
    }

    /// Returns model of some server type.
    fn model(&self, server_type: &ServerType) -> &SwitchingCost {
        &self.0[&server_type.key]
    }
}
