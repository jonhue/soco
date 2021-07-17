//! Switching cost model.

use crate::model::data_center::model::ServerType;
use std::collections::HashMap;

/// Switching cost model.

pub struct SwitchingCostModel(HashMap<String, SwitchingCost>);

/// Switching cost.
pub struct SwitchingCost {
    /// Average cost per unit of energy.
    energy_cost: f64,
    /// Power consumed when idling.
    phi_min: f64,
    /// Power consumed at full load.
    phi_max: f64,
    /// Additional energy consumed by toggling a server on and off.
    epsilon: f64,
    /// Required time in time slots for migrating connections or data.
    delta: f64,
    /// Wear-and-tear costs of toggling a server.
    tau: f64,
    /// Perceived risk associated with toggling a server.
    rho: f64,
}

impl SwitchingCostModel {
    /// Computes switching cost for a server of some type.
    pub fn switching_cost(&self, server_type: &ServerType) -> f64 {
        let model = self.model(server_type);
        model.energy_cost * (model.epsilon + model.delta * model.phi_max)
            + model.tau
            + model.rho
    }

    /// Computes normalized switching cost for a server of some type. Approximately,
    /// measures the minimum duration a server must be asleep to outweigh the switching cost.
    /// Referred to as `\xi` in the paper.
    pub fn normalized_switching_cost(&self, server_type: &ServerType) -> f64 {
        let model = self.model(server_type);
        self.switching_cost(server_type) / (model.energy_cost * model.phi_min)
    }

    /// Builds vector of switching costs for all server types.
    pub fn switching_costs(&self, server_types: &Vec<ServerType>) -> Vec<f64> {
        server_types
            .iter()
            .map(|server_type| self.switching_cost(server_type))
            .collect()
    }

    /// Builds vector of normalized switching costs for all server types.
    pub fn normalized_switching_costs(
        &self,
        server_types: &Vec<ServerType>,
    ) -> Vec<f64> {
        server_types
            .iter()
            .map(|server_type| self.switching_cost(server_type))
            .collect()
    }

    /// Returns model of some server type.
    fn model(&self, server_type: &ServerType) -> &SwitchingCost {
        &self.0[&server_type.key]
    }
}
