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
}

impl SwitchingCostModel {
    pub fn new(switching_costs: HashMap<String, SwitchingCost>) -> Self {
        SwitchingCostModel(switching_costs)
    }

    /// Computes switching cost for a server of some type.
    pub fn switching_cost(&self, server_type: &ServerType) -> N64 {
        let model = self.model(server_type);
        n64(
            model.energy_cost * (model.epsilon + model.delta * model.phi_max)
                + model.tau
                + model.rho,
        )
    }

    /// Computes normalized switching cost for a server of some type. Approximately,
    /// measures the minimum duration a server must be asleep to outweigh the switching cost.
    /// Referred to as `\xi` in the paper.
    pub fn normalized_switching_cost(&self, server_type: &ServerType) -> N64 {
        let model = self.model(server_type);
        self.switching_cost(server_type) / (model.energy_cost * model.phi_min)
    }

    /// Builds vector of switching costs for all server types.
    pub fn switching_costs(&self, server_types: &Vec<ServerType>) -> Vec<f64> {
        server_types
            .iter()
            .map(|server_type| self.switching_cost(server_type).raw())
            .collect()
    }

    /// Builds vector of normalized switching costs for all server types.
    pub fn normalized_switching_costs(
        &self,
        server_types: &Vec<ServerType>,
    ) -> Vec<f64> {
        server_types
            .iter()
            .map(|server_type| self.switching_cost(server_type).raw())
            .collect()
    }

    /// Returns model of some server type.
    fn model(&self, server_type: &ServerType) -> &SwitchingCost {
        &self.0[&server_type.key]
    }
}
