//! Energy cost model.

use crate::model::data_center::model::Location;
use crate::utils::min;
use crate::utils::pos;
use noisy_float::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

/// Energy source.
#[derive(Clone)]
pub struct EnergySource<'a> {
    /// Average cost of a unit of energy during time slot `t`.
    cost: Arc<dyn Fn(i32) -> f64 + Send + Sync + 'a>,
    /// Average profit of an unused unit of energy during time slot `t`.
    profit: Arc<dyn Fn(i32) -> f64 + Send + Sync + 'a>,
    /// Maximum amount of energy at some location during time slot `t`.
    limit: Arc<dyn Fn(i32, &Location) -> f64 + Send + Sync + 'a>,
}
impl<'a> EnergySource<'a> {
    fn cost(&self, t: i32) -> R64 {
        r64((self.cost)(t))
    }
    fn profit(&self, t: i32) -> R64 {
        r64((self.profit)(t))
    }
    fn limit(&self, t: i32, location: &Location) -> R64 {
        r64((self.limit)(t, location))
    }
}

/// Energy cost model. Parameters are provided separately for each location.
#[derive(Clone)]
pub enum EnergyCostModel<'a> {
    /// Linear energy cost.
    Linear(HashMap<String, LinearEnergyCostModel<'a>>),
    /// Energy cost model using (maximum) quotas.
    /// Maximum profit across all energy sources must not exceed overall energy cost.
    Quotas(HashMap<String, QuotasEnergyCostModel<'a>>),
}

#[derive(Clone)]
pub struct LinearEnergyCostModel<'a> {
    /// Average cost of a unit of energy during time slot `t`.
    pub cost: Arc<dyn Fn(i32) -> f64 + Send + Sync + 'a>,
}
impl<'a> LinearEnergyCostModel<'a> {
    fn cost(&self, t: i32) -> R64 {
        r64((self.cost)(t))
    }
}

#[derive(Clone)]
pub struct QuotasEnergyCostModel<'a> {
    /// Energy sources.
    pub sources: Vec<EnergySource<'a>>,
}

impl<'a> EnergyCostModel<'a> {
    /// Energy cost at some location during time slot `t` with energy consumption `p`.
    /// Referred to as `\nu` in the paper.
    pub fn cost(&self, t: i32, location: &Location, p: R64) -> R64 {
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

                let mut result = r64(0.);
                let mut cum_limit = r64(0.);
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
