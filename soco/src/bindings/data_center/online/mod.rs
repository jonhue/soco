use crate::{
    bindings::{utils::make_package, DataCenterCost},
    model::data_center::{
        DataCenterModelOutputFailure, DataCenterModelOutputSuccess,
    },
    problem::{
        FractionalSimplifiedSmoothedConvexOptimization,
        FractionalSmoothedConvexOptimization,
        IntegralSimplifiedSmoothedConvexOptimization,
    },
    streaming::online,
};
use pyo3::prelude::*;

mod lazy_capacity_provisioning;
mod memoryless;
mod probabilistic;
mod randomized;
mod randomly_biased_greedy;

/// Obtained result, integral result, and last memory.
type Response<T, M> = (
    (Vec<Vec<T>>, DataCenterCost),
    (Vec<Vec<i32>>, DataCenterCost),
    Option<M>,
);
/// Obtained result, integral result, and last memory.
type StepResponse<T, M> = (
    (Vec<T>, DataCenterCost),
    (Vec<i32>, DataCenterCost),
    Option<M>,
);

type DataCenterFractionalSmoothedConvexOptimization<'a> =
    FractionalSmoothedConvexOptimization<
        'a,
        DataCenterModelOutputSuccess,
        DataCenterModelOutputFailure,
    >;
type DataCenterFractionalSimplifiedSmoothedConvexOptimization<'a> =
    FractionalSimplifiedSmoothedConvexOptimization<
        'a,
        DataCenterModelOutputSuccess,
        DataCenterModelOutputFailure,
    >;
type DataCenterIntegralSimplifiedSmoothedConvexOptimization<'a> =
    IntegralSimplifiedSmoothedConvexOptimization<
        'a,
        DataCenterModelOutputSuccess,
        DataCenterModelOutputFailure,
    >;

/// Stops backend server.
#[pyfunction]
fn stop(addr: &str) -> PyResult<()> {
    online::stop(addr.parse().unwrap());
    Ok(())
}

pub fn submodule(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(stop, m)?)?;

    let lazy_capacity_provisioning =
        PyModule::new(py, "lazy_capacity_provisioning")?;
    lazy_capacity_provisioning::submodule(py, lazy_capacity_provisioning)?;
    make_package(
        py,
        lazy_capacity_provisioning,
        "soco.data_center.online.lazy_capacity_provisioning",
    );
    m.add_submodule(lazy_capacity_provisioning)?;

    let memoryless = PyModule::new(py, "memoryless")?;
    memoryless::submodule(py, memoryless)?;
    make_package(py, memoryless, "soco.data_center.online.memoryless");
    m.add_submodule(memoryless)?;

    let probabilistic = PyModule::new(py, "probabilistic")?;
    probabilistic::submodule(py, probabilistic)?;
    make_package(py, probabilistic, "soco.data_center.online.probabilistic");
    m.add_submodule(probabilistic)?;

    let randomized = PyModule::new(py, "randomized")?;
    randomized::submodule(py, randomized)?;
    make_package(py, randomized, "soco.data_center.online.randomized");
    m.add_submodule(randomized)?;

    let randomly_biased_greedy = PyModule::new(py, "randomly_biased_greedy")?;
    randomly_biased_greedy::submodule(py, randomly_biased_greedy)?;
    make_package(
        py,
        randomly_biased_greedy,
        "soco.data_center.online.randomly_biased_greedy",
    );
    m.add_submodule(randomly_biased_greedy)?;

    Ok(())
}
