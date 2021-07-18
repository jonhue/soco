//! Model of a data center.

use crate::config::Config;
use crate::cost::{CostFn, SingleCostFn};
use crate::model::data_center::loads::{
    apply_loads_over_time, apply_predicted_loads, LoadFractions, LoadProfile,
};
use crate::model::data_center::models::delay::DelayModel;
use crate::model::data_center::models::energy_consumption::EnergyConsumptionModel;
use crate::model::data_center::models::energy_cost::EnergyCostModel;
use crate::model::data_center::models::revenue_loss::RevenueLossModel;
use crate::model::data_center::models::switching_cost::SwitchingCostModel;
use crate::model::data_center::safe_balancing;
use crate::model::{verify_update, Model, OfflineInput, OnlineInput};
use crate::problem::{
    Online, SimplifiedSmoothedConvexOptimization,
    SmoothedBalancedLoadOptimization, SmoothedConvexOptimization,
    SmoothedLoadOptimization,
};
use crate::value::Value;
use crate::vec_wrapper::VecWrapper;
use noisy_float::prelude::*;
use num::NumCast;
use serde_derive::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

pub static DEFAULT_KEY: &str = "";

/// Server type.
#[derive(Clone)]
pub struct ServerType {
    /// Name.
    pub key: String,
    /// Maximum allowed utilization. Between `0` and `1`.
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
    fn limit_utilization(&self, s: R64, f: impl Fn() -> R64) -> R64 {
        if s <= r64(self.max_utilization) {
            f()
        } else {
            r64(f64::INFINITY)
        }
    }
}

/// Job type.
#[derive(Clone)]
pub struct JobType<'a> {
    /// Name.
    pub key: String,
    /// Processing time `\eta_{k,i}` of a job on some server type `k` (assuming full utilization). Must be less than the time slot length `delta`.
    processing_time_on: Arc<dyn Fn(&ServerType) -> f64 + Send + Sync + 'a>,
}
impl Default for JobType<'_> {
    fn default() -> Self {
        JobType {
            key: DEFAULT_KEY.to_string(),
            processing_time_on: Arc::new(|_| 1.),
        }
    }
}
impl<'a> JobType<'a> {
    pub fn new(processing_times: HashMap<String, f64>) -> Self {
        JobType {
            key: DEFAULT_KEY.to_string(),
            processing_time_on: Arc::new(move |server_type| {
                processing_times[&server_type.key]
            }),
        }
    }

    pub fn processing_time_on(&self, server_type: &ServerType) -> R64 {
        r64((self.processing_time_on)(server_type))
    }
}

/// Geographical source of jobs.
#[derive(Clone)]
pub struct Source<'a> {
    /// Name.
    pub key: String,
    /// Routing delay `\delta_{t,j,s}` to location `j` during time slot `t`.
    routing_delay_to: Arc<dyn Fn(i32, &Location) -> f64 + Send + Sync + 'a>,
}
impl<'a> Source<'a> {
    pub fn routing_delay_to(&self, t: i32, location: &Location) -> R64 {
        r64((self.routing_delay_to)(t, location))
    }
}
impl Default for Source<'_> {
    fn default() -> Self {
        Source {
            key: DEFAULT_KEY.to_string(),
            routing_delay_to: Arc::new(|_, _| 0.),
        }
    }
}

/// Data center.
#[derive(Clone)]
pub struct Location {
    /// Name.
    pub key: String,
    /// Maximum number of servers of each type.
    pub m: HashMap<String, i32>,
}

/// Model of a network of data centers.
#[derive(Clone)]
pub struct DataCenterModel<'a> {
    /// Length of a time slot.
    delta: f64,
    /// Weight of revenue loss (as opposed to energy cost). `gamma > 0`.
    gamma: f64,
    /// Locations.
    locations: Vec<Location>,
    /// Server types.
    server_types: Vec<ServerType>,
    /// Sources, i.e. geographically centered locations.
    sources: Vec<Source<'a>>,
    /// Job types.
    job_types: Vec<JobType<'a>>,
    /// Energy consumption model.
    energy_consumption_model: EnergyConsumptionModel,
    /// Energy cost model.
    energy_cost_model: EnergyCostModel<'a>,
    /// Revenue loss model.
    revenue_loss_model: RevenueLossModel,
    /// Delay model.
    delay_model: DelayModel,
    /// Switching cost model.
    switching_cost_model: SwitchingCostModel,
}

impl<'a> DataCenterModel<'a> {
    /// Creates a model of a singular data center.
    #[allow(clippy::too_many_arguments)]
    pub fn single(
        delta: f64,
        gamma: f64,
        server_types: Vec<ServerType>,
        m: HashMap<String, i32>,
        job_types: Vec<JobType<'a>>,
        energy_consumption_model: EnergyConsumptionModel,
        energy_cost_model: EnergyCostModel<'a>,
        revenue_loss_model: RevenueLossModel,
        delay_model: DelayModel,
        switching_cost_model: SwitchingCostModel,
    ) -> Self {
        Self {
            delta,
            gamma,
            locations: vec![Location {
                key: "".to_string(),
                m,
            }],
            server_types,
            sources: vec![Source::default()],
            job_types,
            energy_consumption_model,
            energy_cost_model,
            revenue_loss_model,
            delay_model,
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
        sources: Vec<Source<'a>>,
        job_types: Vec<JobType<'a>>,
        energy_consumption_model: EnergyConsumptionModel,
        energy_cost_model: EnergyCostModel<'a>,
        revenue_loss_model: RevenueLossModel,
        delay_model: DelayModel,
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
            delay_model,
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
    ) -> R64 {
        loads
            .iter()
            .enumerate()
            .map(|(i_, &load)| {
                let (_, i) = parse(self.job_types.len(), i_);
                let processing_time =
                    self.job_types[i].processing_time_on(server_type);
                processing_time * load
            })
            .sum::<R64>()
    }

    /// Energy cost. Non-negative convex operating cost of data center `j`
    /// during time slot `t` with configuration `x` load profile `lambda` and load fractions `zs`.
    /// Referred to as `e` in the paper.
    fn energy_cost<T>(
        &self,
        t: i32,
        j: usize,
        x: &Config<T>,
        lambda: &LoadProfile,
        zs: &LoadFractions,
    ) -> R64
    where
        T: Value<'a>,
    {
        let p = self.energy_consumption(j, x, lambda, zs);
        self.energy_cost_model.cost(t, &self.locations[j], p)
    }

    /// Energy consumption of data center `j` with configuration `x_`, load profile
    /// `lambda`, and load fractions `zs`.
    /// Referred to as `\phi'` in the paper.
    fn energy_consumption<T>(
        &self,
        j: usize,
        x_: &Config<T>,
        lambda: &LoadProfile,
        zs: &LoadFractions,
    ) -> R64
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
    /// Referred to as `q` in the paper.
    fn revenue_loss(
        &self,
        t: i32,
        location: &Location,
        server_type: &ServerType,
        source: &Source,
        job_type: &JobType,
        l: R64,
    ) -> R64 {
        let delay = self.delay_model.average_delay(self.delta, l)
            + source.routing_delay_to(t, location)
            + job_type.processing_time_on(server_type);
        r64(self.gamma) * (self.revenue_loss_model.loss(t, job_type, delay))
    }

    /// Revenue loss across all sources and job types.
    /// Referred to as `h` in the paper.
    fn overall_revenue_loss<T>(
        &self,
        t: i32,
        location: &Location,
        server_type: &ServerType,
        x_: T,
        loads: &LoadProfile,
    ) -> R64
    where
        T: Value<'a>,
    {
        let x = NumCast::from(x_).unwrap();
        let total_load = self.total_sub_jobs(server_type, loads);
        safe_balancing(x, total_load, || {
            (0..self.sources.len())
                .map(|s| -> R64 {
                    (0..self.job_types.len())
                        .map(|i| {
                            r64(loads[encode(self.job_types.len(), s, i)])
                                * self.revenue_loss(
                                    t,
                                    location,
                                    server_type,
                                    &self.sources[s],
                                    &self.job_types[i],
                                    total_load / x,
                                )
                        })
                        .sum()
                })
                .sum()
        })
    }

    /// Objective to be minimized when assigning jobs.
    /// Referred to as `f` in the paper.
    fn objective<T>(
        &self,
        t: i32,
        x: &Config<T>,
        lambda: &LoadProfile,
        zs: &LoadFractions,
    ) -> R64
    where
        T: Value<'a>,
    {
        (0..self.locations.len())
            .map(|j| -> R64 {
                self.energy_cost(t, j, x, lambda, zs)
                    + (0..self.server_types.len())
                        .map(|k| -> R64 {
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
                        .sum::<R64>()
            })
            .sum::<R64>()
    }

    /// Optimally applies (certain) loads to the model of a data center to obtain a cost function.
    /// Referred to as `f` in the paper.
    ///
    /// * `loads` - vector of loads for all time slots that should be supported by the returned cost function
    /// * `t_start` - time offset, i.e. time of first load profile
    fn apply_loads_over_time<T>(
        &'a self,
        loads: Vec<LoadProfile>,
        t_start: i32,
    ) -> CostFn<'a, Config<T>>
    where
        T: Value<'a>,
    {
        apply_loads_over_time(
            self.d_(),
            self.e_(),
            move |t, x, lambda, zs| self.objective(t, x, lambda, zs),
            loads,
            t_start,
        )
    }

    /// Optimally applies loads from a single to the model of a data center to obtain a cost function.
    /// Referred to as `f` in the paper.
    ///
    /// * `loads` - a load profile for each predicted sample (one load profile for certainty) over the supported time horizon
    /// * `t_start` - time offset, i.e. time of first load samples
    fn apply_predicted_loads<T>(
        &'a self,
        loads: Vec<Vec<LoadProfile>>,
        t_start: i32,
    ) -> SingleCostFn<'a, Config<T>>
    where
        T: Value<'a>,
    {
        apply_predicted_loads(
            self.d_(),
            self.e_(),
            move |t, x, lambda, zs| self.objective(t, x, lambda, zs),
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
    fn generate_bounds<T>(&self) -> Vec<T>
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

#[derive(Debug)]
pub struct DataCenterOfflineInput {
    /// Vector of loads for all time slots that should be supported by the returned cost function.
    pub loads: Vec<LoadProfile>,
}
impl Default for DataCenterOfflineInput {
    fn default() -> Self {
        DataCenterOfflineInput { loads: vec![] }
    }
}
impl OfflineInput for DataCenterOfflineInput {}

#[derive(Debug, Deserialize, Serialize)]
pub struct DataCenterOnlineInput {
    /// A load profile for each predicted sample (one load profile for certainty) over the supported time horizon.
    pub loads: Vec<Vec<LoadProfile>>,
}
impl<'a> OnlineInput<'a> for DataCenterOnlineInput {}

impl<'a, T>
    Model<
        'a,
        SmoothedConvexOptimization<'a, T>,
        DataCenterOfflineInput,
        DataCenterOnlineInput,
    > for DataCenterModel<'a>
where
    T: Value<'a>,
{
    fn to(
        &'a self,
        input: DataCenterOfflineInput,
    ) -> SmoothedConvexOptimization<'a, T> {
        let ssco_p: SimplifiedSmoothedConvexOptimization<'a, T> =
            self.to(input);
        ssco_p.into_sco()
    }

    fn update(
        &'a self,
        o: &mut Online<SmoothedConvexOptimization<'a, T>>,
        DataCenterOnlineInput { loads }: DataCenterOnlineInput,
    ) {
        verify_update(o, loads.len() as i32);
        let t_start = o.p.hitting_cost.now() + 1;
        println!("Updating online instance to time slot {}.", t_start);
        o.p.hitting_cost
            .add(self.apply_predicted_loads(loads, t_start))
    }
}

impl<'a, T>
    Model<
        'a,
        SimplifiedSmoothedConvexOptimization<'a, T>,
        DataCenterOfflineInput,
        DataCenterOnlineInput,
    > for DataCenterModel<'a>
where
    T: Value<'a>,
{
    fn to(
        &'a self,
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
        &'a self,
        o: &mut Online<SimplifiedSmoothedConvexOptimization<'a, T>>,
        DataCenterOnlineInput { loads }: DataCenterOnlineInput,
    ) {
        verify_update(o, loads.len() as i32);
        let t_start = o.p.hitting_cost.now() + 1;
        println!("Updating online instance to time slot {}.", t_start);
        o.p.hitting_cost
            .add(self.apply_predicted_loads(loads, t_start))
    }
}

impl<'a, T>
    Model<
        'a,
        SmoothedBalancedLoadOptimization<'a, T>,
        DataCenterOfflineInput,
        DataCenterOnlineInput,
    > for DataCenterModel<'a>
where
    T: Value<'a>,
{
    /// Notes:
    /// * Only allows for a single location, source, and job type.
    fn to(
        &'a self,
        DataCenterOfflineInput { loads }: DataCenterOfflineInput,
    ) -> SmoothedBalancedLoadOptimization<'a, T> {
        assert!(self.locations.len() == 1);
        let location = &self.locations[0];
        assert!(self.sources.len() == 1);
        let source = &self.sources[0];
        assert!(self.job_types.len() == 1);
        let job_type = &self.job_types[0];

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
                SingleCostFn::certain(move |t, l_| {
                    let l = r64(l_);
                    let s = l / self.delta;
                    let p = server_type.limit_utilization(s, || {
                        self.energy_consumption_model
                            .consumption(server_type, s)
                    });
                    let energy_cost =
                        self.energy_cost_model.cost(t, location, p);
                    let revenue_loss = self.revenue_loss(
                        t,
                        location,
                        server_type,
                        source,
                        job_type,
                        l,
                    );
                    energy_cost + revenue_loss
                })
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
        &'a self,
        o: &mut Online<SmoothedBalancedLoadOptimization<'a, T>>,
        DataCenterOnlineInput { loads }: DataCenterOnlineInput,
    ) {
        verify_update(o, loads.len() as i32);
        println!(
            "Updating online instance to time slot {}.",
            o.p.load.len() + 1
        );
        for load_profiles in loads {
            assert!(load_profiles.len() == 1);
            o.p.load.push(NumCast::from(load_profiles[0][0]).unwrap());
        }
    }
}

impl<'a, T>
    Model<
        'a,
        SmoothedLoadOptimization<T>,
        DataCenterOfflineInput,
        DataCenterOnlineInput,
    > for DataCenterModel<'a>
where
    T: Value<'a>,
{
    /// Notes:
    /// * Only allows for a single location, source, and job type.
    /// * Assumes full utilization and averages the energy cost over the time horizon.
    fn to(
        &'a self,
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
                            .consumption(server_type, r64(1.));
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
        &'a self,
        o: &mut Online<SmoothedLoadOptimization<T>>,
        DataCenterOnlineInput { loads }: DataCenterOnlineInput,
    ) {
        verify_update(o, loads.len() as i32);
        println!(
            "Updating online instance to time slot {}.",
            o.p.load.len() + 1
        );
        for load_profiles in loads {
            assert!(load_profiles.len() == 1);
            o.p.load.push(NumCast::from(load_profiles[0][0]).unwrap());
        }
    }
}
