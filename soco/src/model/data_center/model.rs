//! Model of a data center.

use crate::config::Config;
use crate::cost::{CostFn, SingleCostFn};
use crate::model::data_center::loads::{
    apply_loads_over_time, apply_predicted_loads, LoadFractions, LoadProfile,
};
use crate::model::data_center::models::delay::average_delay;
use crate::model::data_center::models::energy_consumption::EnergyConsumptionModel;
use crate::model::data_center::models::energy_cost::EnergyCostModel;
use crate::model::data_center::models::revenue_loss::RevenueLossModel;
use crate::model::data_center::models::switching_cost::SwitchingCostModel;
use crate::model::data_center::safe_balancing;
use crate::model::{verify_update, Model, OfflineInput, OnlineInput};
use crate::problem::{
    Online, Problem, SimplifiedSmoothedConvexOptimization,
    SmoothedBalancedLoadOptimization, SmoothedConvexOptimization,
    SmoothedLoadOptimization,
};
use crate::value::Value;
use crate::vec_wrapper::VecWrapper;
use noisy_float::prelude::*;
use num::NumCast;
use pyo3::prelude::*;
use serde_derive::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

pub static DEFAULT_KEY: &str = "";

/// Server type.
#[pyclass]
#[derive(Clone)]
pub struct ServerType {
    /// Name.
    #[pyo3(get, set)]
    pub key: String,
    /// Maximum allowed utilization. Between `0` and `1`.
    #[pyo3(get, set)]
    max_utilization: f64,
}
impl Default for ServerType {
    fn default() -> Self {
        ServerType {
            key: DEFAULT_KEY.to_string(),
            max_utilization: 1.,
        }
    }
}
impl ServerType {
    fn limit_utilization(&self, s: N64, f: impl Fn() -> N64) -> N64 {
        if s <= n64(self.max_utilization) {
            f()
        } else {
            n64(f64::INFINITY)
        }
    }
}
#[pymethods]
impl ServerType {
    #[new]
    fn constructor(key: String, max_utilization: f64) -> Self {
        ServerType {
            key,
            max_utilization,
        }
    }
}

/// Job type.
#[pyclass]
#[derive(Clone)]
pub struct JobType {
    /// Name.
    #[pyo3(get, set)]
    pub key: String,
    /// Processing time `\eta_{k,i}` of a job on some server type `k` (assuming full utilization). Must be less than the time slot length `delta`.
    processing_time_on: Arc<dyn Fn(&ServerType) -> f64 + Send + Sync>,
}
impl Default for JobType {
    fn default() -> Self {
        JobType {
            key: DEFAULT_KEY.to_string(),
            processing_time_on: Arc::new(|_| 1.),
        }
    }
}
impl JobType {
    pub fn from_map(processing_times: HashMap<String, f64>) -> Self {
        JobType {
            key: DEFAULT_KEY.to_string(),
            processing_time_on: Arc::new(move |server_type| {
                processing_times[&server_type.key]
            }),
        }
    }

    pub fn processing_time_on(&self, server_type: &ServerType) -> N64 {
        n64((self.processing_time_on)(server_type))
    }
}
#[pymethods]
impl JobType {
    #[new]
    fn constructor(key: String, processing_time_on: Py<PyAny>) -> Self {
        JobType {
            key,
            processing_time_on: Arc::new(move |server_type| {
                Python::with_gil(|py| {
                    processing_time_on
                        .call1(py, (server_type.clone(),))
                        .unwrap()
                        .extract(py)
                        .unwrap()
                })
            }),
        }
    }
}

/// Geographical source of jobs.
#[pyclass]
#[derive(Clone)]
pub struct Source {
    /// Name.
    #[pyo3(get, set)]
    pub key: String,
    /// Routing delay `\delta_{t,j,s}` to location `j` during time slot `t`.
    routing_delay_to: Arc<dyn Fn(i32, &Location) -> f64 + Send + Sync>,
}
impl Source {
    pub fn routing_delay_to(&self, t: i32, location: &Location) -> N64 {
        n64((self.routing_delay_to)(t, location))
    }
}
impl Default for Source {
    fn default() -> Self {
        Source {
            key: DEFAULT_KEY.to_string(),
            routing_delay_to: Arc::new(|_, _| 0.),
        }
    }
}
#[pymethods]
impl Source {
    #[new]
    fn constructor(key: String, routing_delay_to: Py<PyAny>) -> Self {
        Source {
            key,
            routing_delay_to: Arc::new(move |t, location| {
                Python::with_gil(|py| {
                    routing_delay_to
                        .call1(py, (t, location.clone()))
                        .unwrap()
                        .extract(py)
                        .unwrap()
                })
            }),
        }
    }
}

/// Data center.
#[pyclass]
#[derive(Clone)]
pub struct Location {
    /// Name.
    #[pyo3(get, set)]
    pub key: String,
    /// Maximum number of servers of each type.
    #[pyo3(get, set)]
    pub m: HashMap<String, i32>,
}
#[pymethods]
impl Location {
    #[new]
    fn constructor(key: String, m: HashMap<String, i32>) -> Self {
        Location { key, m }
    }
}

/// Model of a network of data centers.
#[pyclass]
#[derive(Clone)]
pub struct DataCenterModel {
    /// Length of a time slot.
    #[pyo3(get, set)]
    delta: f64,
    /// Weight of revenue loss (as opposed to energy cost). `gamma >= 0`.
    #[pyo3(get, set)]
    gamma: f64,
    /// Locations.
    #[pyo3(get, set)]
    locations: Vec<Location>,
    /// Server types.
    #[pyo3(get, set)]
    server_types: Vec<ServerType>,
    /// Sources, i.e. geographically centered locations.
    #[pyo3(get, set)]
    sources: Vec<Source>,
    /// Job types.
    #[pyo3(get, set)]
    job_types: Vec<JobType>,
    /// Energy consumption model.
    #[pyo3(set)]
    energy_consumption_model: EnergyConsumptionModel,
    /// Energy cost model.
    #[pyo3(set)]
    energy_cost_model: EnergyCostModel,
    /// Revenue loss model.
    #[pyo3(set)]
    revenue_loss_model: RevenueLossModel,
    /// Switching cost model.
    #[pyo3(set)]
    switching_cost_model: SwitchingCostModel,
}

#[pymethods]
impl DataCenterModel {
    /// Creates a new model.
    #[allow(clippy::too_many_arguments)]
    #[new]
    pub fn new(
        delta: f64,
        gamma: f64,
        locations: Vec<Location>,
        server_types: Vec<ServerType>,
        sources: Vec<Source>,
        job_types: Vec<JobType>,
        energy_consumption_model: EnergyConsumptionModel,
        energy_cost_model: EnergyCostModel,
        revenue_loss_model: RevenueLossModel,
        switching_cost_model: SwitchingCostModel,
    ) -> Self {
        Self {
            delta,
            gamma,
            locations,
            server_types,
            sources,
            job_types,
            energy_consumption_model,
            energy_cost_model,
            revenue_loss_model,
            switching_cost_model,
        }
    }
}

impl DataCenterModel {
    /// Creates a model of a singular data center.
    #[allow(clippy::too_many_arguments)]
    pub fn single(
        delta: f64,
        gamma: f64,
        server_types: Vec<ServerType>,
        m: HashMap<String, i32>,
        job_types: Vec<JobType>,
        energy_consumption_model: EnergyConsumptionModel,
        energy_cost_model: EnergyCostModel,
        revenue_loss_model: RevenueLossModel,
        switching_cost_model: SwitchingCostModel,
    ) -> Self {
        Self {
            delta,
            gamma,
            locations: vec![Location {
                key: DEFAULT_KEY.to_string(),
                m,
            }],
            server_types,
            sources: vec![Source::default()],
            job_types,
            energy_consumption_model,
            energy_cost_model,
            revenue_loss_model,
            switching_cost_model,
        }
    }

    /// Creates a model of a network of data centers.
    #[allow(clippy::too_many_arguments)]
    pub fn network(
        delta: f64,
        gamma: f64,
        locations: Vec<Location>,
        server_types: Vec<ServerType>,
        sources: Vec<Source>,
        job_types: Vec<JobType>,
        energy_consumption_model: EnergyConsumptionModel,
        energy_cost_model: EnergyCostModel,
        revenue_loss_model: RevenueLossModel,
        switching_cost_model: SwitchingCostModel,
    ) -> Self {
        Self {
            delta,
            gamma,
            locations,
            server_types,
            sources,
            job_types,
            energy_consumption_model,
            energy_cost_model,
            revenue_loss_model,
            switching_cost_model,
        }
    }

    /// Calculates cumulative sub jobs of servers of some type, i.e. the number
    /// of sub jobs handled by all servers of this type, when they are assigned
    /// the load profile `loads`.
    fn total_sub_jobs(
        &self,
        server_type: &ServerType,
        loads: &LoadProfile,
    ) -> N64 {
        loads
            .iter()
            .enumerate()
            .map(|(i_, &load)| {
                let (_, i) = parse(self.job_types.len(), i_);
                let processing_time =
                    self.job_types[i].processing_time_on(server_type);
                processing_time * load
            })
            .sum()
    }

    /// Energy cost. Non-negative convex operating cost of data center `j`
    /// during time slot `t` with configuration `x` load profile `lambda` and load fractions `zs`.
    /// Referred to as `e` in the paper.
    fn energy_cost<'a, T>(
        &self,
        t: i32,
        j: usize,
        x: &Config<T>,
        lambda: &LoadProfile,
        zs: &LoadFractions,
    ) -> N64
    where
        T: Value<'a>,
    {
        let p = self.energy_consumption(j, x, lambda, zs);
        self.energy_cost_model.cost(t, &self.locations[j], p)
    }

    /// Energy consumption of data center `j` with configuration `x_`, load profile
    /// `lambda`, and load fractions `zs`.
    /// Referred to as `\phi'` in the paper.
    fn energy_consumption<'a, T>(
        &self,
        j: usize,
        x_: &Config<T>,
        lambda: &LoadProfile,
        zs: &LoadFractions,
    ) -> N64
    where
        T: Value<'a>,
    {
        (0..self.server_types.len())
            .map(|k| {
                let k_ = encode(self.server_types.len(), j, k);
                let server_type = &self.server_types[k];
                let total_load = self
                    .total_sub_jobs(server_type, &zs.select_loads(lambda, k_));
                let x = NumCast::from(x_[k_]).unwrap();
                safe_balancing(x, total_load, || {
                    let s = total_load / (x * self.delta);
                    server_type.limit_utilization(s, || {
                        x * self
                            .energy_consumption_model
                            .consumption(server_type, s)
                    })
                })
            })
            .sum()
    }

    /// Revenue loss. Non-negative convex cost incurred by processing some job
    /// on some server during time slot `t` when a total of `l` sub jobs are
    /// processed on the server.
    /// `number_of_jobs` is the number of jobs processed on the server and
    /// `mean_job_duration` is their mean duration.
    /// Referred to as `q` in the paper.
    #[allow(clippy::too_many_arguments)]
    fn revenue_loss(
        &self,
        t: i32,
        location: &Location,
        server_type: &ServerType,
        source: &Source,
        job_type: &JobType,
        number_of_jobs: N64,
        mean_job_duration: N64,
    ) -> N64 {
        let delay =
            average_delay(self.delta, number_of_jobs, mean_job_duration)
                + source.routing_delay_to(t, location)
                + job_type.processing_time_on(server_type);
        if delay.is_infinite() {
            n64(f64::INFINITY)
        } else {
            n64(self.gamma) * self.revenue_loss_model.loss(t, job_type, delay)
        }
    }

    /// Revenue loss across all sources and job types.
    /// Referred to as `h` in the paper.
    fn overall_revenue_loss<'a, T>(
        &self,
        t: i32,
        location: &Location,
        server_type: &ServerType,
        x_: T,
        loads: &LoadProfile,
    ) -> N64
    where
        T: Value<'a>,
    {
        let x = NumCast::from(x_).unwrap();
        let total_load = self.total_sub_jobs(server_type, loads);

        // calculates the mean duration of jobs on a server of some type under the load profile `loads`
        let number_of_jobs = loads.iter().sum();
        let mean_job_duration = if number_of_jobs == n64(0.) {
            n64(0.)
        } else {
            total_load / number_of_jobs
        };

        safe_balancing(x, number_of_jobs, || {
            (0..self.sources.len())
                .map(|s| -> N64 {
                    (0..self.job_types.len())
                        .map(|i| {
                            loads[encode(self.job_types.len(), s, i)]
                                * self.revenue_loss(
                                    t,
                                    location,
                                    server_type,
                                    &self.sources[s],
                                    &self.job_types[i],
                                    number_of_jobs / x,
                                    mean_job_duration,
                                )
                        })
                        .sum()
                })
                .sum()
        })
    }

    /// Objective to be minimized when assigning jobs.
    /// Referred to as `f` in the paper.
    fn objective<'a, T>(
        &self,
        t: i32,
        x: &Config<T>,
        lambda: &LoadProfile,
        zs: &LoadFractions,
    ) -> N64
    where
        T: Value<'a>,
    {
        (0..self.locations.len())
            .map(|j| -> N64 {
                self.energy_cost(t, j, x, lambda, zs)
                    + (0..self.server_types.len())
                        .map(|k| -> N64 {
                            let k_ = encode(self.server_types.len(), j, k);
                            let loads = zs.select_loads(lambda, k_);
                            self.overall_revenue_loss(
                                t,
                                &self.locations[j],
                                &self.server_types[k],
                                x[k_],
                                &loads,
                            )
                        })
                        .sum::<N64>()
            })
            .sum::<N64>()
    }

    /// Optimally applies (certain) loads to the model of a data center to obtain a cost function.
    /// Referred to as `f` in the paper.
    ///
    /// * `loads` - vector of loads for all time slots that should be supported by the returned cost function
    /// * `t_start` - time offset, i.e. time of first load profile
    fn apply_loads_over_time<'a, T>(
        &self,
        loads: Vec<LoadProfile>,
        t_start: i32,
    ) -> CostFn<'a, Config<T>>
    where
        T: Value<'a>,
    {
        let model = self.clone();
        apply_loads_over_time(
            self.d_(),
            self.e_(),
            move |t, x, lambda, zs| model.objective(t, x, lambda, zs),
            loads,
            t_start,
        )
    }

    /// Optimally applies loads from a single to the model of a data center to obtain a cost function.
    /// Referred to as `f` in the paper.
    ///
    /// * `loads` - a load profile for each predicted sample (one load profile for certainty) over the supported time horizon
    /// * `t_start` - time offset, i.e. time of first load samples
    fn apply_predicted_loads<'a, T>(
        &self,
        loads: Vec<Vec<LoadProfile>>,
        t_start: i32,
    ) -> SingleCostFn<'a, Config<T>>
    where
        T: Value<'a>,
    {
        let model = self.clone();
        apply_predicted_loads(
            self.d_(),
            self.e_(),
            move |t, x, lambda, zs| model.objective(t, x, lambda, zs),
            loads,
            t_start,
        )
    }

    /// Number of dimensions of the underlying problem.
    pub fn d_(&self) -> i32 {
        (self.locations.len() * self.server_types.len()) as i32
    }

    /// Number of load types of the underlying problem.
    pub fn e_(&self) -> i32 {
        (self.sources.len() * self.job_types.len()) as i32
    }

    /// Generates upper bounds of the underlying problem instance.
    fn generate_bounds<'a, T>(&self) -> Vec<T>
    where
        T: Value<'a>,
    {
        (0..self.d_() as usize)
            .map(|k_| {
                let (j, k) = parse(self.server_types.len(), k_);
                NumCast::from(self.locations[j].m[&self.server_types[k].key])
                    .unwrap()
            })
            .collect()
    }
}

/// Parses index of underlying representation, returns outer and inner indexes.
fn parse(inner_len: usize, i: usize) -> (usize, usize) {
    let outer = i / inner_len;
    let inner = i - outer * inner_len;
    (outer, inner)
}

/// Encodes index of underlying representation from the outer and inner indexes.
fn encode(inner_len: usize, outer: usize, inner: usize) -> usize {
    outer * inner_len + inner
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct DataCenterOfflineInput {
    /// Vector of loads for all time slots that should be supported by the returned cost function.
    #[pyo3(get, set)]
    pub loads: Vec<LoadProfile>,
}
impl Default for DataCenterOfflineInput {
    fn default() -> Self {
        DataCenterOfflineInput { loads: vec![] }
    }
}
impl OfflineInput for DataCenterOfflineInput {}
#[pymethods]
impl DataCenterOfflineInput {
    #[new]
    fn constructor(loads: Vec<LoadProfile>) -> Self {
        DataCenterOfflineInput { loads }
    }
}

#[pyclass]
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DataCenterOnlineInput {
    /// A load profile for each predicted sample (one load profile for certainty) over the supported time horizon.
    #[pyo3(get, set)]
    pub loads: Vec<Vec<LoadProfile>>,
}
impl OnlineInput for DataCenterOnlineInput {}
#[pymethods]
impl DataCenterOnlineInput {
    #[new]
    fn constructor(loads: Vec<Vec<LoadProfile>>) -> Self {
        DataCenterOnlineInput { loads }
    }
}

impl<'a, T>
    Model<
        SmoothedConvexOptimization<'a, T>,
        DataCenterOfflineInput,
        DataCenterOnlineInput,
    > for DataCenterModel
where
    T: Value<'a>,
{
    fn to(
        &self,
        input: DataCenterOfflineInput,
    ) -> SmoothedConvexOptimization<'a, T> {
        let ssco_p: SimplifiedSmoothedConvexOptimization<'a, T> =
            self.to(input);
        ssco_p.into_sco()
    }

    fn update(
        &self,
        o: &mut Online<SmoothedConvexOptimization<'a, T>>,
        DataCenterOnlineInput { loads }: DataCenterOnlineInput,
    ) {
        o.p.inc_t_end();
        let t = o.p.t_end();
        let span = loads.len() as i32;
        println!("Updating online instance to time slot {}.", t);
        o.p.hitting_cost
            .add(t, self.apply_predicted_loads(loads, t));
        verify_update(o, span);
    }
}

impl<'a, T>
    Model<
        SimplifiedSmoothedConvexOptimization<'a, T>,
        DataCenterOfflineInput,
        DataCenterOnlineInput,
    > for DataCenterModel
where
    T: Value<'a>,
{
    fn to(
        &self,
        DataCenterOfflineInput { loads }: DataCenterOfflineInput,
    ) -> SimplifiedSmoothedConvexOptimization<'a, T> {
        let d = self.d_();
        let t_end = loads.len() as i32;
        let bounds = self.generate_bounds();
        let switching_cost = self
            .switching_cost_model
            .switching_costs(&self.server_types);
        let hitting_cost = self.apply_loads_over_time(loads, 1);
        SimplifiedSmoothedConvexOptimization {
            d,
            t_end,
            bounds,
            switching_cost,
            hitting_cost,
        }
    }

    fn update(
        &self,
        o: &mut Online<SimplifiedSmoothedConvexOptimization<'a, T>>,
        DataCenterOnlineInput { loads }: DataCenterOnlineInput,
    ) {
        o.p.inc_t_end();
        let t = o.p.t_end();
        let span = loads.len() as i32;
        println!("Updating online instance to time slot {}.", t);
        o.p.hitting_cost
            .add(t, self.apply_predicted_loads(loads, t));
        verify_update(o, span);
    }
}

impl<'a, T>
    Model<
        SmoothedBalancedLoadOptimization<'a, T>,
        DataCenterOfflineInput,
        DataCenterOnlineInput,
    > for DataCenterModel
where
    T: Value<'a>,
{
    /// Notes:
    /// * Only allows for a single location, source, and job type.
    fn to(
        &self,
        DataCenterOfflineInput { loads }: DataCenterOfflineInput,
    ) -> SmoothedBalancedLoadOptimization<'a, T> {
        assert!(self.locations.len() == 1);
        let location = &self.locations[0];
        assert!(self.sources.len() == 1);
        assert!(self.job_types.len() == 1);

        let d = self.d_();
        let t_end = loads.len() as i32;
        let bounds = self.generate_bounds();
        let switching_cost = self
            .switching_cost_model
            .switching_costs(&self.server_types);
        let hitting_cost = self
            .server_types
            .clone()
            .into_iter()
            .map(move |server_type| {
                let location = location.clone();
                let energy_consumption_model =
                    self.energy_consumption_model.clone();
                let energy_cost_model = self.energy_cost_model.clone();
                let delta = self.delta;
                CostFn::new(
                    1,
                    SingleCostFn::certain(move |t, l| {
                        let s = n64(l) / delta;
                        let p = server_type.limit_utilization(s, || {
                            energy_consumption_model
                                .consumption(&server_type, s)
                        });
                        energy_cost_model.cost(t, &location, p)
                    }),
                )
            })
            .collect();
        let load = loads
            .iter()
            .map(|lambda| NumCast::from(lambda[0]).unwrap())
            .collect();
        SmoothedBalancedLoadOptimization {
            d,
            t_end,
            bounds,
            switching_cost,
            hitting_cost,
            load,
        }
    }

    fn update(
        &self,
        o: &mut Online<SmoothedBalancedLoadOptimization<'a, T>>,
        DataCenterOnlineInput { loads }: DataCenterOnlineInput,
    ) {
        o.p.inc_t_end();
        let t = o.p.t_end();
        assert!(t == o.p.load.len() as i32 + 1, "Loads and time slot are inconsistent. Time slot is {} but loads are present for {} time slots.", t, o.p.load.len() as i32 + 1);
        let span = loads.len() as i32;
        println!("Updating online instance to time slot {}.", t);
        for load_profiles in loads {
            assert!(
                load_profiles.len() == 1,
                "Load profiles for SBLO need to be homogeneous and certain."
            );
            o.p.load.push(NumCast::from(load_profiles[0][0]).unwrap());
        }
        verify_update(o, span);
    }
}

impl<'a, T>
    Model<
        SmoothedLoadOptimization<T>,
        DataCenterOfflineInput,
        DataCenterOnlineInput,
    > for DataCenterModel
where
    T: Value<'a>,
{
    /// Notes:
    /// * Only allows for a single location, source, and job type.
    /// * Assumes full utilization and averages the energy cost over the time horizon.
    fn to(
        &self,
        DataCenterOfflineInput { loads }: DataCenterOfflineInput,
    ) -> SmoothedLoadOptimization<T> {
        assert!(self.locations.len() == 1);
        let location = &self.locations[0];
        assert!(self.sources.len() == 1);
        assert!(self.job_types.len() == 1);

        let d = self.d_();
        let t_end = loads.len() as i32;
        let bounds = self.generate_bounds();
        let switching_cost = self
            .switching_cost_model
            .switching_costs(&self.server_types);
        let hitting_cost = self
            .server_types
            .iter()
            .map(move |server_type| {
                (0..t_end)
                    .map(|t| -> f64 {
                        let p = self
                            .energy_consumption_model
                            .consumption(server_type, n64(1.));
                        self.energy_cost_model.cost(t, location, p).raw()
                    })
                    .sum::<f64>()
                    / t_end as f64
            })
            .collect();
        let load = loads
            .iter()
            .map(|lambda| NumCast::from(lambda[0]).unwrap())
            .collect();
        SmoothedLoadOptimization {
            d,
            t_end,
            bounds,
            switching_cost,
            hitting_cost,
            load,
        }
    }

    fn update(
        &self,
        o: &mut Online<SmoothedLoadOptimization<T>>,
        DataCenterOnlineInput { loads }: DataCenterOnlineInput,
    ) {
        o.p.inc_t_end();
        let t = o.p.t_end();
        assert!(t == o.p.load.len() as i32 + 1, "Loads and time slot are inconsistent. Time slot is {} but loads are present for {} time slots.", t, o.p.load.len() as i32 + 1);
        let span = loads.len() as i32;
        println!("Updating online instance to time slot {}.", t);
        for load_profiles in loads {
            assert!(load_profiles.len() == 1);
            o.p.load.push(NumCast::from(load_profiles[0][0]).unwrap());
        }
        verify_update(o, span);
    }
}
