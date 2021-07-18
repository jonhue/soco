//! Definition of load profiles.

use crate::config::Config;
use crate::cost::{CostFn, SingleCostFn};
use crate::numerics::convex_optimization::{minimize, Constraint};
use crate::utils::{access, shift_time, unshift_time};
use crate::value::Value;
use crate::vec_wrapper::VecWrapper;
use serde_derive::{Deserialize, Serialize};
use std::iter::FromIterator;
use std::ops::Div;
use std::ops::Index;
use std::ops::Mul;
use std::sync::Arc;

/// For some time `t`, encapsulates the load of `e` types.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct LoadProfile(Vec<f64>);

impl LoadProfile {
    /// Creates a new load profile from a vector.
    pub fn new(l: Vec<f64>) -> LoadProfile {
        LoadProfile(l)
    }

    /// Creates a load profile with a single load type.
    pub fn single(l: f64) -> LoadProfile {
        LoadProfile(vec![l])
    }

    /// Number of load types.
    pub fn e(&self) -> i32 {
        self.0.len() as i32
    }

    /// Sum of loads across all load types.
    pub fn total(&self) -> f64 {
        self.0.iter().copied().sum()
    }

    /// Converts load profile to a vector.
    pub fn to_vec(&self) -> Vec<f64> {
        self.0.clone()
    }
}

impl Index<usize> for LoadProfile {
    type Output = f64;

    fn index(&self, i: usize) -> &Self::Output {
        assert!(
            i < self.0.len(),
            "argument must denote one of {} types, is {}",
            self.0.len(),
            i + 1
        );
        &self.0[i]
    }
}

impl VecWrapper for LoadProfile {
    type Item = f64;

    fn to_vec(&self) -> &Vec<Self::Item> {
        &self.0
    }
}

impl FromIterator<f64> for LoadProfile {
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = f64>,
    {
        let mut l = vec![];
        for j in iter {
            l.push(j);
        }
        LoadProfile::new(l)
    }
}

impl Mul<Vec<f64>> for LoadProfile {
    type Output = LoadProfile;

    /// Applies fractions to loads.
    fn mul(self, z: Vec<f64>) -> Self::Output {
        self.iter()
            .zip(&z)
            .map(|(&l, &z)| l * z)
            .collect::<LoadProfile>()
    }
}

impl Div<f64> for LoadProfile {
    type Output = LoadProfile;

    /// Divides loads by scalar.
    fn div(self, other: f64) -> Self::Output {
        self.iter().map(|&l| l / other).collect()
    }
}

/// Assignment of load fractions to dimensions for each job type.
pub struct LoadFractions<'a> {
    /// Stores for each dimension `k in [d]` the fractions of loads `i in [e]` that are handled.
    /// Flat representation where position `k * e + i` represents the fraction of jobs of type `i` assigned to servers of type `k`.
    zs_: &'a [f64],
    /// Number of dimensions.
    d: i32,
    /// Number of job types.
    e: i32,
}

impl LoadFractions<'_> {
    /// Returns the load fraction for dimension `k` and job type `i`.
    pub fn get(&self, k: usize, i: usize) -> f64 {
        assert!(k < self.d as usize);
        assert!(i < self.e as usize);
        self.zs_[k * self.e as usize + i]
    }

    /// Selects loads for dimension `k`.
    pub fn select_loads(&self, lambda: &LoadProfile, k: usize) -> LoadProfile {
        self.zs_[k * self.e as usize..k * self.e as usize + self.e as usize]
            .iter()
            .enumerate()
            .map(|(i, z)| lambda[i] * z)
            .collect()
    }
}

/// Optimally applies (certain) loads to a model to obtain a cost function.
///
/// * `d` - number of dimensions
/// * `e` - number of job types
/// * `objective` - cost function to minimize w.r.t. load assignments
/// * `loads` - vector of (certain) loads for all time slots that should be supported by the returned cost function
/// * `t_start` - time offset, i.e. time of first load profile
pub fn apply_loads_over_time<'a, T>(
    d: i32,
    e: i32,
    objective: impl Fn(i32, &Config<T>, &LoadProfile, &LoadFractions) -> f64
        + Send
        + Sync
        + 'a,
    loads: Vec<LoadProfile>,
    t_start: i32,
) -> CostFn<'a, Config<T>>
where
    T: Value<'a>,
{
    CostFn::stretch(
        t_start,
        shift_time(loads.len() as i32, t_start),
        SingleCostFn::certain(move |t, x: Config<T>| {
            let lambda = access(&loads, unshift_time(t, t_start)).unwrap();
            apply_loads(d, e, &objective, lambda, t, x)
        }),
    )
}

/// Optimally applies loads to a model to obtain a cost function.
///
/// * `d` - number of dimensions
/// * `e` - number of job types
/// * `objective` - cost function to minimize w.r.t. load assignments
/// * `loads` - a load profile for each predicted sample (one load profile for certainty) over the supported time horizon
/// * `t_start` - time offset, i.e. time of first load samples
pub fn apply_predicted_loads<'a, T>(
    d: i32,
    e: i32,
    objective: impl Fn(i32, &Config<T>, &LoadProfile, &LoadFractions) -> f64
        + Send
        + Sync
        + 'a,
    loads: Vec<Vec<LoadProfile>>,
    t_start: i32,
) -> SingleCostFn<'a, Config<T>>
where
    T: Value<'a>,
{
    SingleCostFn::predictive(move |t, x: Config<T>| {
        access(&loads, unshift_time(t, t_start))
            .unwrap()
            .iter()
            .map(|lambda| apply_loads(d, e, &objective, lambda, t, x.clone()))
            .collect()
    })
}

/// Calculates cost based on a model for an optimal distribution of loads.
///
/// * `d` - number of dimensions
/// * `e` - number of job types
/// * `objective` - cost function to minimize w.r.t. load assignments
/// * `lambda` - load profile
/// * `t` - time slot
/// * `x` - configuration
pub fn apply_loads<'a, T>(
    d: i32,
    e: i32,
    objective: &impl Fn(i32, &Config<T>, &LoadProfile, &LoadFractions) -> f64,
    lambda: &LoadProfile,
    t: i32,
    x: Config<T>,
) -> f64
where
    T: Value<'a>,
{
    assert!(e == lambda.e());

    // we store for each dimension `k in [d]` the fractions of loads `i in [e]` that are handled
    let solver_d = (d * e) as usize;
    let bounds = vec![(0., 1.); solver_d];
    let objective = |zs_: &[f64]| {
        let zs = LoadFractions { zs_, d, e };
        objective(t, &x, lambda, &zs)
    };

    // assigns each dimension a fraction of each load type
    let init = vec![1. / solver_d as f64; solver_d];

    // ensure that the fractions across all dimensions of each load type sum to `1`
    // let equality_constraints = vec![];
    let equality_constraints = (0..e as usize)
        .map(|i| -> Constraint {
            let lambda = lambda.clone();
            Arc::new(move |zs_: &[f64]| -> f64 {
                let total_lambda = lambda.total();
                if total_lambda > 0. {
                    let zs = LoadFractions { zs_, d, e };
                    (0..d as usize).map(|k| zs.get(k, i)).sum::<f64>()
                        - lambda[i] / total_lambda
                } else {
                    0.
                }
            })
        })
        .collect();

    // minimize cost across all possible server to load matchings
    let (_, opt) =
        minimize(objective, &bounds, Some(init), vec![], equality_constraints)
            .unwrap();
    opt
}
