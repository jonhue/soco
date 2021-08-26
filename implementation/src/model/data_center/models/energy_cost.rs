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
#[pymethods]
impl EnergySource {
    #[new]
    fn constructor(
        cost: Py<PyAny>,
        profit: Py<PyAny>,
        limit: Py<PyAny>,
    ) -> Self {
        EnergySource {
            cost: Arc::new(move |t| {
                Python::with_gil(|py| {
                    cost.call1(py, (t,))
                        .expect("energy source `cost` method invalid")
                        .extract(py)
                        .expect("energy source `cost` method invalid")
                })
            }),
            profit: Arc::new(move |t| {
                Python::with_gil(|py| {
                    profit
                        .call1(py, (t,))
                        .expect("energy source `profit` method invalid")
                        .extract(py)
                        .expect("energy source `profit` method invalid")
                })
            }),
            limit: Arc::new(move |t, location| {
                Python::with_gil(|py| {
                    limit
                        .call1(py, (t, location.clone()))
                        .expect("energy source `limit` method invalid")
                        .extract(py)
                        .expect("energy source `limit` method invalid")
                })
            }),
        }
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
#[pymethods]
impl LinearEnergyCostModel {
    #[new]
    fn constructor(cost: Py<PyAny>) -> Self {
        LinearEnergyCostModel {
            cost: Arc::new(move |t| {
                Python::with_gil(|py| {
                    cost.call1(py, (t,))
                        .expect(
                            "linear energy cost model `cost` method invalid",
                        )
                        .extract(py)
                        .expect(
                            "linear energy cost model `cost` method invalid",
                        )
                })
            }),
        }
    }

    #[staticmethod]
    pub fn from_const(cost: f64) -> Self {
        LinearEnergyCostModel {
            cost: Arc::new(move |_t| cost),
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct QuotasEnergyCostModel {
    /// Energy sources.
    #[pyo3(get, set)]
    pub sources: Vec<EnergySource>,
}
#[pymethods]
impl QuotasEnergyCostModel {
    #[new]
    fn constructor(sources: Vec<EnergySource>) -> Self {
        QuotasEnergyCostModel { sources }
    }
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
