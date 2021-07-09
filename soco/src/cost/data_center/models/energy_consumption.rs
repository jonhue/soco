//! Energy consumption model.

use crate::cost::data_center::model::ServerType;
use std::collections::HashMap;

/// Energy consumption model.
pub enum EnergyConsumptionModel {
    /// Linear model from Dayarathna et al.
    Linear(HashMap<String, Linear>),
    /// Simplification of the linear model with the assumption that servers
    /// consume half of their peak power when idling.
    SimplifiedLinear(HashMap<String, SimplifiedLinear>),
    /// Non-linear model from Dayarathna et al.
    NonLinear(HashMap<String, NonLinear>),
}

pub struct Linear {
    /// Power consumed when idling.
    phi_min: f64,
    /// Power consumed at full load.
    phi_max: f64,
}

pub struct SimplifiedLinear {
    /// Power consumed at full load.
    phi_max: f64,
}

pub struct NonLinear {
    /// Power consumed when idling.
    phi_min: f64,
    /// Constant for computing dynamic power. `alpha > 1`.
    alpha: f64,
    /// Constant for computing dynamic power. `beta > 0`.
    beta: f64,
}

impl EnergyConsumptionModel {
    /// Energy consumption of a server of some type with utilization `s`.
    /// Referred to as `\phi` in the paper.
    pub fn consumption(&self, server_type: &ServerType, s: f64) -> f64 {
        match self {
            EnergyConsumptionModel::Linear(models) => {
                let model = &models[&server_type.key];
                (model.phi_max - model.phi_min) * s + model.phi_min
            }
            EnergyConsumptionModel::SimplifiedLinear(models) => {
                let model = &models[&server_type.key];
                model.phi_max * (1. + s) / 2.
            }
            EnergyConsumptionModel::NonLinear(models) => {
                let model = &models[&server_type.key];
                s.powf(model.alpha) / model.beta + model.phi_min
            }
        }
    }
}
