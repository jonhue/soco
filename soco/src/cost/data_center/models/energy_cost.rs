//! Energy cost model.

use crate::cost::data_center::model::Location;
use crate::utils::min;
use crate::utils::pos;
use std::collections::HashMap;
use std::sync::Arc;

/// Energy source.
#[derive(Clone)]
pub struct EnergySource<'a> {
    /// Average cost of a unit of energy during time slot `t`.
    cost: Arc<dyn Fn(i32) -> f64 + 'a>,
    /// Average profit of an unused unit of energy during time slot `t`.
    profit: Arc<dyn Fn(i32) -> f64 + 'a>,
    /// Maximum amount of energy at some location during time slot `t`.
    limit: Arc<dyn Fn(i32, &Location) -> f64 + 'a>,
}
impl<'a> EnergySource<'a> {
    fn cost(&self, t: i32) -> f64 {
        (self.cost)(t)
    }
    fn profit(&self, t: i32) -> f64 {
        (self.profit)(t)
    }
    fn limit(&self, t: i32, location: &Location) -> f64 {
        (self.limit)(t, location)
    }
}

/// Energy cost model.
pub enum EnergyCostModel<'a> {
    /// Linear energy cost.
    Linear(HashMap<String, Linear<'a>>),
    /// Energy cost model using (maximum) quotas.
    /// Maximum profit across all energy sources must not exceed overall energy cost.
    Quotas(HashMap<String, Quotas<'a>>),
}

pub struct Linear<'a> {
    /// Average cost of a unit of energy during time slot `t`.
    cost: Arc<dyn Fn(i32) -> f64 + 'a>,
}
impl<'a> Linear<'a> {
    fn cost(&self, t: i32) -> f64 {
        (self.cost)(t)
    }
}

pub struct Quotas<'a> {
    /// Energy sources.
    sources: Vec<EnergySource<'a>>,
}

impl<'a> EnergyCostModel<'a> {
    /// Energy cost at some location during time slot `t` with energy consumption `p`.
    /// Referred to as `\nu` in the paper.
    pub fn cost(&self, t: i32, location: &Location, p: f64) -> f64 {
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

                let mut result = 0.;
                let mut cum_limit = 0.;
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
