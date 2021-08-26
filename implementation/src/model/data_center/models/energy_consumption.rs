//! Energy consumption model.

use crate::model::data_center::model::ServerType;
use noisy_float::prelude::*;
use pyo3::prelude::*;
use std::collections::HashMap;

/// Energy consumption model. Parameters are provided separately for each server type.
#[derive(Clone, FromPyObject)]
pub enum EnergyConsumptionModel {
    /// Linear model from Dayarathna et al.
    Linear(HashMap<String, LinearEnergyConsumptionModel>),
    /// Simplification of the linear model with the assumption that servers
    /// consume half of their peak power when idling.
    SimplifiedLinear(HashMap<String, SimplifiedLinearEnergyConsumptionModel>),
    /// Non-linear model from Dayarathna et al.
    NonLinear(HashMap<String, NonLinearEnergyConsumptionModel>),
}

#[pyclass]
#[derive(Clone)]
pub struct LinearEnergyConsumptionModel {
    /// Power consumed when idling in a unit of time.
    #[pyo3(get, set)]
    pub phi_min: f64,
    /// Power consumed at full load in a unit of time.
    #[pyo3(get, set)]
    pub phi_max: f64,
}
#[pymethods]
impl LinearEnergyConsumptionModel {
    #[new]
    fn constructor(phi_min: f64, phi_max: f64) -> Self {
        LinearEnergyConsumptionModel { phi_min, phi_max }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct SimplifiedLinearEnergyConsumptionModel {
    /// Power consumed at full load in a unit of time.
    #[pyo3(get, set)]
    pub phi_max: f64,
}
#[pymethods]
impl SimplifiedLinearEnergyConsumptionModel {
    #[new]
    fn constructor(phi_max: f64) -> Self {
        SimplifiedLinearEnergyConsumptionModel { phi_max }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct NonLinearEnergyConsumptionModel {
    /// Power consumed when idling in a unit of time.
    #[pyo3(get, set)]
    pub phi_min: f64,
    /// Constant for computing dynamic power. `alpha > 1`.
    #[pyo3(get, set)]
    pub alpha: f64,
    /// Constant for computing dynamic power. `beta > 0`.
    #[pyo3(get, set)]
    pub beta: f64,
}
#[pymethods]
impl NonLinearEnergyConsumptionModel {
    #[new]
    fn constructor(phi_min: f64, alpha: f64, beta: f64) -> Self {
        NonLinearEnergyConsumptionModel {
            phi_min,
            alpha,
            beta,
        }
    }
}

impl EnergyConsumptionModel {
    /// Energy consumption of a server of some type with utilization `s`.
    /// Referred to as $\phi$ in the paper.
    pub fn consumption(
        &self,
        delta: f64,
        server_type: &ServerType,
        s: N64,
    ) -> N64 {
        match self {
            EnergyConsumptionModel::Linear(models) => {
                let model = &models[&server_type.key];
                n64(delta)
                    * (n64(model.phi_max - model.phi_min) * s
                        + n64(model.phi_min))
            }
            EnergyConsumptionModel::SimplifiedLinear(models) => {
                let model = &models[&server_type.key];
                n64(delta) * (n64(model.phi_max) * (n64(1.) + s) / n64(2.))
            }
            EnergyConsumptionModel::NonLinear(models) => {
                let model = &models[&server_type.key];
                n64(delta)
                    * (s.powf(n64(model.alpha)) / n64(model.beta)
                        + n64(model.phi_min))
            }
        }
    }
}
