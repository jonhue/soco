//! Energy consumption model.

use crate::model::data_center::model::ServerType;
use std::collections::HashMap;

/// Energy consumption model. Parameters are provided separately for each server type.
#[derive(Clone)]
pub enum EnergyConsumptionModel {
    /// Linear model from Dayarathna et al.
    Linear(HashMap<String, LinearEnergyConsumptionModel>),
    /// Simplification of the linear model with the assumption that servers
    /// consume half of their peak power when idling.
    SimplifiedLinear(HashMap<String, SimplifiedLinearEnergyConsumptionModel>),
    /// Non-linear model from Dayarathna et al.
    NonLinear(HashMap<String, NonLinearEnergyConsumptionModel>),
}

#[derive(Clone)]
pub struct LinearEnergyConsumptionModel {
    /// Power consumed when idling.
    pub phi_min: f64,
    /// Power consumed at full load.
    pub phi_max: f64,
}

#[derive(Clone)]
pub struct SimplifiedLinearEnergyConsumptionModel {
    /// Power consumed at full load.
    pub phi_max: f64,
}

#[derive(Clone)]
pub struct NonLinearEnergyConsumptionModel {
    /// Power consumed when idling.
    pub phi_min: f64,
    /// Constant for computing dynamic power. `alpha > 1`.
    pub alpha: f64,
    /// Constant for computing dynamic power. `beta > 0`.
    pub beta: f64,
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
