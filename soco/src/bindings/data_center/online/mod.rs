use crate::{
    bindings::{utils::make_package, DataCenterCost},
    cost::Cost,
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

mod multi_dimensional;
mod uni_dimensional;

/// Obtained result, integral result, and last memory.
type Response<T, M> = (
    (Vec<Vec<T>>, DataCenterCost),
    (Vec<Vec<i32>>, DataCenterCost),
    Option<M>,
    u128,
);
type SLOResponse<T, M> = (
    (Vec<Vec<T>>, Cost<(), DataCenterModelOutputFailure>),
    (Vec<Vec<i32>>, Cost<(), DataCenterModelOutputFailure>),
    Option<M>,
    u128,
);
/// Obtained result, integral result, and last memory.
type StepResponse<T, M> = (
    (Vec<T>, DataCenterCost),
    (Vec<i32>, DataCenterCost),
    Option<M>,
    u128,
);
type SLOStepResponse<T, M> = (
    (Vec<T>, Cost<(), DataCenterModelOutputFailure>),
    (Vec<i32>, Cost<(), DataCenterModelOutputFailure>),
    Option<M>,
    u128,
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

    let multi_dimensional = PyModule::new(py, "multi_dimensional")?;
    multi_dimensional::submodule(py, multi_dimensional)?;
    make_package(
        py,
        multi_dimensional,
        "soco.data_center.online.multi_dimensional",
    );
    m.add_submodule(multi_dimensional)?;

    let uni_dimensional = PyModule::new(py, "uni_dimensional")?;
    uni_dimensional::submodule(py, uni_dimensional)?;
    make_package(
        py,
        uni_dimensional,
        "soco.data_center.online.uni_dimensional",
    );
    m.add_submodule(uni_dimensional)?;

    Ok(())
}
