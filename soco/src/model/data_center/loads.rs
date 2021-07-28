//! Definition of load profiles.

use crate::config::Config;
use crate::cost::{CostFn, SingleCostFn};
use crate::numerics::convex_optimization::{minimize, WrappedObjective};
use crate::numerics::ApplicablePrecision;
use crate::utils::{access, transpose, unshift_time};
use crate::value::Value;
use crate::vec_wrapper::VecWrapper;
use log::debug;
use noisy_float::prelude::*;
use num::NumCast;
use pyo3::prelude::*;
use rand::prelude::IteratorRandom;
use rand::thread_rng;
use rayon::iter::{
    FromParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    ParallelIterator,
};
use rayon::slice::Iter;
use rayon::vec::IntoIter;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::cmp::{max, min, Ordering};
use std::iter::FromIterator;
use std::ops::Div;
use std::ops::Index;
use std::ops::Mul;

static MAX_SAMPLE_SIZE: i32 = 100;

/// For some time `t`, encapsulates the load of `e` types.
#[derive(Clone, Debug, PartialEq)]
pub struct LoadProfile(Vec<N64>);

impl LoadProfile {
    /// Creates a new load profile from a raw vector.
    pub fn raw(l: Vec<f64>) -> LoadProfile {
        LoadProfile(l.iter().map(|&z| n64(z)).collect())
    }

    /// Creates a new load profile from a vector.
    pub fn new(l: Vec<N64>) -> LoadProfile {
        LoadProfile(l)
    }

    /// Creates a load profile with a single load type.
    pub fn single(l: N64) -> LoadProfile {
        LoadProfile(vec![l])
    }

    /// Number of load types.
    pub fn e(&self) -> i32 {
        self.0.len() as i32
    }

    /// Sum of loads across all load types.
    pub fn total(&self) -> N64 {
        self.0.iter().copied().sum()
    }

    /// Converts load profile to a vector.
    pub fn to_vec(&self) -> Vec<N64> {
        self.0.clone()
    }

    /// Converts load profile to a vector.
    pub fn to_raw(&self) -> Vec<f64> {
        self.0.iter().map(|z| z.raw()).collect()
    }
}

impl<'a> FromPyObject<'a> for LoadProfile {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        Ok(LoadProfile::raw(ob.extract()?))
    }
}

impl IntoPy<PyObject> for LoadProfile {
    fn into_py(self, py: Python) -> PyObject {
        self.to_raw().into_py(py)
    }
}

impl Serialize for LoadProfile {
    #[inline]
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.to_raw().serialize(s)
    }
}

impl<'a> Deserialize<'a> for LoadProfile {
    #[inline]
    fn deserialize<D: Deserializer<'a>>(d: D) -> Result<Self, D::Error> {
        Vec::<f64>::deserialize(d).map(LoadProfile::raw)
    }
}

impl Index<usize> for LoadProfile {
    type Output = N64;

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
    type Item = N64;

    fn to_vec(&self) -> &Vec<Self::Item> {
        &self.0
    }
}

impl FromIterator<N64> for LoadProfile {
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = N64>,
    {
        LoadProfile::new(Vec::<N64>::from_iter(iter))
    }
}

impl FromParallelIterator<N64> for LoadProfile {
    fn from_par_iter<I>(iter: I) -> Self
    where
        I: IntoParallelIterator<Item = N64>,
    {
        LoadProfile::new(Vec::<N64>::from_par_iter(iter))
    }
}

impl<'a> IntoParallelIterator for &'a LoadProfile {
    type Item = &'a N64;
    type Iter = Iter<'a, N64>;

    fn into_par_iter(self) -> Self::Iter {
        self.0.par_iter()
    }
}

impl IntoParallelIterator for LoadProfile {
    type Item = N64;
    type Iter = IntoIter<N64>;

    fn into_par_iter(self) -> Self::Iter {
        self.0.into_par_iter()
    }
}

impl Mul<Vec<N64>> for LoadProfile {
    type Output = LoadProfile;

    /// Applies fractions to loads.
    fn mul(self, z: Vec<N64>) -> Self::Output {
        self.iter()
            .zip(&z)
            .map(|(&l, &z)| l * z)
            .collect::<LoadProfile>()
    }
}

impl Div<N64> for LoadProfile {
    type Output = LoadProfile;

    /// Divides loads by scalar.
    fn div(self, other: N64) -> Self::Output {
        self.iter().map(|&l| l / other).collect()
    }
}

/// For some time `t`, encapsulates the load of `e` types as multiple samples per type.
#[derive(Clone, Debug)]
pub struct PredictedLoadProfile(Vec<Vec<N64>>);

impl PredictedLoadProfile {
    /// Creates a new predicted load profile from a raw vector of vectors.
    pub fn raw(l: Vec<Vec<f64>>) -> PredictedLoadProfile {
        PredictedLoadProfile(
            l.iter()
                .map(|zs| zs.iter().map(|&z| n64(z)).collect())
                .collect(),
        )
    }

    /// Creates a new predicted load profile from a vector of vectors.
    pub fn new(l: Vec<Vec<N64>>) -> PredictedLoadProfile {
        PredictedLoadProfile(l)
    }

    /// Number of load types.
    pub fn e(&self) -> i32 {
        self.0.len() as i32
    }

    /// Smallest sample size for some job type.
    fn smallest_sample_size(&self) -> i32 {
        self.0.iter().map(|zs| zs.len()).min().unwrap() as i32
    }

    /// Largest sample size for some job type.
    fn largest_sample_size(&self) -> i32 {
        self.0.iter().map(|zs| zs.len()).max().unwrap() as i32
    }

    /// Converts predicted load profile to a vector.
    pub fn to_vec(&self) -> Vec<Vec<N64>> {
        self.0.clone()
    }

    /// Converts predicted load profile to a vector.
    pub fn to_raw(&self) -> Vec<Vec<f64>> {
        self.0
            .iter()
            .map(|zs| zs.iter().map(|z| z.raw()).collect())
            .collect()
    }

    /// Samples load profiles.
    pub fn sample_load_profiles(&self) -> Vec<LoadProfile> {
        let mut rng = thread_rng();
        let sample_size = min(
            max(self.smallest_sample_size(), MAX_SAMPLE_SIZE),
            self.largest_sample_size(),
        );

        // we only use a randomly chosen subset of all samples to remain efficient
        let samples = self
            .to_vec()
            .into_iter()
            .map(|zs| {
                assert!(zs.len() >= sample_size as usize);
                zs.into_iter()
                    .choose_multiple(&mut rng, sample_size as usize)
            })
            .collect();
        transpose(samples)
            .into_iter()
            .map(LoadProfile::new)
            .collect()
    }
}

impl<'a> FromPyObject<'a> for PredictedLoadProfile {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        Ok(PredictedLoadProfile::raw(ob.extract()?))
    }
}

impl IntoPy<PyObject> for PredictedLoadProfile {
    fn into_py(self, py: Python) -> PyObject {
        self.to_raw().into_py(py)
    }
}

impl Serialize for PredictedLoadProfile {
    #[inline]
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.to_raw().serialize(s)
    }
}

impl<'a> Deserialize<'a> for PredictedLoadProfile {
    #[inline]
    fn deserialize<D: Deserializer<'a>>(d: D) -> Result<Self, D::Error> {
        Vec::<Vec<f64>>::deserialize(d).map(PredictedLoadProfile::raw)
    }
}

impl Index<usize> for PredictedLoadProfile {
    type Output = Vec<N64>;

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

impl VecWrapper for PredictedLoadProfile {
    type Item = Vec<N64>;

    fn to_vec(&self) -> &Vec<Self::Item> {
        &self.0
    }
}

impl FromIterator<Vec<N64>> for PredictedLoadProfile {
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = Vec<N64>>,
    {
        PredictedLoadProfile::new(Vec::<Vec<N64>>::from_iter(iter))
    }
}

impl FromParallelIterator<Vec<N64>> for PredictedLoadProfile {
    fn from_par_iter<I>(iter: I) -> Self
    where
        I: IntoParallelIterator<Item = Vec<N64>>,
    {
        PredictedLoadProfile::new(Vec::<Vec<N64>>::from_par_iter(iter))
    }
}

impl<'a> IntoParallelIterator for &'a PredictedLoadProfile {
    type Item = &'a Vec<N64>;
    type Iter = Iter<'a, Vec<N64>>;

    fn into_par_iter(self) -> Self::Iter {
        self.0.par_iter()
    }
}

impl IntoParallelIterator for PredictedLoadProfile {
    type Item = Vec<N64>;
    type Iter = IntoIter<Vec<N64>>;

    fn into_par_iter(self) -> Self::Iter {
        self.0.into_par_iter()
    }
}

/// Assignment of load fractions to dimensions for each job type.
#[derive(Debug)]
pub struct LoadFractions {
    /// Stores for each dimension `k in [d]` the fractions of loads `i in [e]` that are handled.
    /// Flat representation where position `k * e + i` represents the fraction of jobs of type `i` assigned to servers of type `k`.
    zs: Vec<N64>,
    /// Number of dimensions.
    d: i32,
    /// Number of job types.
    e: i32,
}

impl LoadFractions {
    /// Returns the load fraction for dimension `k` and job type `i`.
    pub fn get(&self, k: usize, i: usize, lambda: &LoadProfile) -> N64 {
        match k.cmp(&(self.d as usize - 1)) {
            Ordering::Less => self.zs[k * self.e as usize + i],
            Ordering::Equal => {
                // computes final dimension based on other dimensions
                let total_lambda = lambda.total();
                if total_lambda > 0. {
                    lambda[i] / total_lambda
                        - (0..k)
                            .into_iter()
                            .map(|j| self.zs[j * self.e as usize + i])
                            .sum::<N64>()
                } else {
                    n64(0.)
                }
            }
            Ordering::Greater => panic!("Invalid dimension."),
        }
    }

    /// Selects loads for dimension `k`.
    pub fn select_loads(&self, lambda: &LoadProfile, k: usize) -> LoadProfile {
        (0..self.e as usize)
            .into_iter()
            .map(|i| lambda[i] * self.get(k, i, lambda))
            .collect()
    }
}
impl LoadFractions {
    fn new(zs_: &[f64], d: i32, e: i32) -> Self {
        LoadFractions {
            zs: zs_.iter().map(|&z| n64(z.apply_precision())).collect(),
            d,
            e,
        }
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
    objective: impl Fn(i32, &Config<T>, &LoadProfile, &LoadFractions) -> N64
        + Send
        + Sync
        + 'a,
    loads: Vec<LoadProfile>,
    t_start: i32,
) -> CostFn<'a, Config<T>>
where
    T: Value<'a>,
{
    CostFn::new(
        t_start,
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
/// * `predicted_loads` - vector of predicted loads for all time slots that should be supported by the returned cost function
/// * `t_start` - time offset, i.e. time of first load samples
pub fn apply_predicted_loads<'a, T>(
    d: i32,
    e: i32,
    objective: impl Fn(i32, &Config<T>, &LoadProfile, &LoadFractions) -> N64
        + Send
        + Sync
        + 'a,
    predicted_loads: Vec<PredictedLoadProfile>,
    t_start: i32,
) -> SingleCostFn<'a, Config<T>>
where
    T: Value<'a>,
{
    SingleCostFn::predictive(move |t, x: Config<T>| {
        debug!("{};{};{}", t, t_start, unshift_time(t, t_start));
        let predicted_load_profile =
            access(&predicted_loads, unshift_time(t, t_start)).unwrap();
        predicted_load_profile
            .sample_load_profiles()
            .into_par_iter()
            .map(|lambda| apply_loads(d, e, &objective, &lambda, t, x.clone()))
            .collect()
    })
}

struct ObjectiveData<T> {
    d: i32,
    e: i32,
    lambda: LoadProfile,
    t: i32,
    x: Config<T>,
}

struct ConstraintData {
    d: i32,
    e: i32,
    i: usize,
    lambda: LoadProfile,
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
    objective: &impl Fn(i32, &Config<T>, &LoadProfile, &LoadFractions) -> N64,
    lambda: &LoadProfile,
    t: i32,
    x: Config<T>,
) -> N64
where
    T: Value<'a>,
{
    assert!(e == lambda.e());

    // we store for each dimension `k in [d]` the fractions of loads `i in [e]` that are handled
    // the final dimensions are completely determined by all preceding dimensions
    let solver_d = (d * e) as usize - e as usize;
    let bounds = vec![(0., 1.); solver_d];
    let solver_objective = WrappedObjective::new(
        ObjectiveData {
            d,
            e,
            lambda: lambda.clone(),
            t,
            x: x.clone(),
        },
        |zs_, data| {
            let zs = LoadFractions::new(zs_, data.d, data.e);
            objective(data.t, &data.x, &data.lambda, &zs)
        },
    );

    // assigns each dimension a fraction of each load type
    // note: the chosen assignment ensures that any server type with `0` active servers
    // is also assigned an initial load fraction of `0`
    let number_of_non_zero_entries =
        x.iter().filter(|&&j| j > NumCast::from(0).unwrap()).count();
    let init = if number_of_non_zero_entries == 0 {
        vec![1. / (solver_d as f64 + e as f64); solver_d]
    } else {
        let value = 1. / (number_of_non_zero_entries as f64 * e as f64);
        let vec_x = x.to_vec();
        let (_, butlast_x) = vec_x.split_last().unwrap();
        butlast_x
            .iter()
            .map(|&j| {
                if j == NumCast::from(0).unwrap() {
                    vec![0.; e as usize]
                } else {
                    vec![value; e as usize]
                }
            })
            .flatten()
            .collect()
    };

    // ensure that the fractions across all solver dimensions of each load type do not exceed `1`
    let constraints = (0..e as usize)
        .map(|i| {
            WrappedObjective::new(
                ConstraintData {
                    d,
                    e,
                    i,
                    lambda: lambda.clone(),
                },
                |zs_, data| {
                    let zs = LoadFractions::new(zs_, data.d, data.e);
                    -zs.get(d as usize - 1, data.i, &data.lambda)
                },
            )
        })
        .collect();

    // minimize cost across all possible server to load matchings
    let (_, opt) =
        minimize(solver_objective, bounds, Some(init), constraints).unwrap();
    opt
}
