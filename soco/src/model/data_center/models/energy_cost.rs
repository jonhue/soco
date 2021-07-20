//! Energy cost model.

use crate::model::data_center::model::Location;
use crate::utils::min;
use crate::utils::pos;
use noisy_float::prelude::*;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

/// Energy source.
#[pyclass]
#[derive(Clone)]
pub struct EnergySource {
    /// Average cost of a unit of energy during time slot `t`.
    cost: Arc<dyn Fn(i32) -> f64 + Send + Sync>,
    /// Average profit of an unused unit of energy during time slot `t`.
    profit: Arc<dyn Fn(i32) -> f64 + Send + Sync>,
    /// Maximum amount of energy at some location during time slot `t`.
    limit: Arc<dyn Fn(i32, &Location) -> f64 + Send + Sync>,
}
impl EnergySource {
    fn cost(&self, t: i32) -> N64 {
        n64((self.cost)(t))
    }
    fn profit(&self, t: i32) -> N64 {
        n64((self.profit)(t))
    }
    fn limit(&self, t: i32, location: &Location) -> N64 {
        n64((self.limit)(t, location))
    }
}

/// Energy cost model. Parameters are provided separately for each location.
#[derive(Clone, FromPyObject)]
pub enum EnergyCostModel {
    /// Linear energy cost.
    Linear(HashMap<String, LinearEnergyCostModel>),
    /// Energy cost model using (maximum) quotas.
    /// Maximum profit across all energy sources must not exceed overall energy cost.
    Quotas(HashMap<String, QuotasEnergyCostModel>),
}

#[pyclass]
#[derive(Clone)]
pub struct LinearEnergyCostModel {
    /// Average cost of a unit of energy during time slot `t`.
    pub cost: Arc<dyn Fn(i32) -> f64 + Send + Sync>,
}
impl LinearEnergyCostModel {
    fn cost(&self, t: i32) -> N64 {
        n64((self.cost)(t))
    }
}

#[pyclass]
#[derive(Clone)]
pub struct QuotasEnergyCostModel {
    /// Energy sources.
    pub sources: Vec<EnergySource>,
}

impl EnergyCostModel {
    /// Energy cost at some location during time slot `t` with energy consumption `p`.
    /// Referred to as `\nu` in the paper.
    pub fn cost(&self, t: i32, location: &Location, p: N64) -> N64 {
        match self {
            EnergyCostModel::Linear(models) => {
                let model = &models[&location.key];
                model.cost(t) * p
            }
            EnergyCostModel::Quotas(models) => {
                let mut sources = models[&location.key].sources.clone();
                sources.sort_by(|a, b| {
                    (a.cost(t) + a.profit(t))
                        .partial_cmp(&(b.cost(t) + b.profit(t)))
                        .unwrap()
                });

                let mut result = n64(0.);
                let mut cum_limit = n64(0.);
                for source in sources {
                    let delta = pos(p - cum_limit);
                    let limit = source.limit(t, location);
                    result += source.cost(t) + min(delta, limit)
                        - source.profit(t) * pos(limit - delta);
                    cum_limit += limit;
                }
                result
            }
        }
    }
}
