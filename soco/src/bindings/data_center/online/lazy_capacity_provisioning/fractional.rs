use crate::{
    algorithms::online::uni_dimensional::lazy_capacity_provisioning::{
        lcp, Memory,
    },
    bindings::data_center::online::{Response, StepResponse},
    model::data_center::model::{
        DataCenterModel, DataCenterModelOutputFailure,
        DataCenterModelOutputSuccess, DataCenterOfflineInput,
        DataCenterOnlineInput,
    },
    problem::FractionalSimplifiedSmoothedConvexOptimization,
    streaming::online::{self, OfflineResponse},
};
use pyo3::prelude::*;

/// Starts backend in a new thread.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn start(
    addr: String,
    model: DataCenterModel,
    input: DataCenterOfflineInput,
    w: i32,
) -> PyResult<Response<f64, Memory<f64>>> {
    let OfflineResponse {
        xs: (xs, cost),
        int_xs: (int_xs, int_cost),
        m,
    } = online::start(
        addr.parse().unwrap(),
        model,
        &lcp::<
            f64,
            FractionalSimplifiedSmoothedConvexOptimization<
                DataCenterModelOutputSuccess,
                DataCenterModelOutputFailure,
            >,
            DataCenterModelOutputSuccess,
            DataCenterModelOutputFailure,
        >,
        (),
        w,
        input,
        None,
    )
    .unwrap();
    Ok(((xs.to_vec(), cost), (int_xs.to_vec(), int_cost), m))
}

/// Executes next iteration of the algorithm.
#[pyfunction]
fn next(
    py: Python,
    addr: String,
    input: DataCenterOnlineInput,
) -> PyResult<StepResponse<f64, Memory<f64>>> {
    py.allow_threads(|| {
        let ((x, cost), (int_x, int_cost), m) =
            online::next::<
                f64,
                FractionalSimplifiedSmoothedConvexOptimization<
                    DataCenterModelOutputSuccess,
                    DataCenterModelOutputFailure,
                >,
                Memory<f64>,
                DataCenterOnlineInput,
                DataCenterModelOutputSuccess,
                DataCenterModelOutputFailure,
            >(addr.parse().unwrap(), input);
        Ok(((x.to_vec(), cost), (int_x.to_vec(), int_cost), m))
    })
}

/// Lazy Capacity Provisioning
pub fn submodule(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(start, m)?)?;
    m.add_function(wrap_pyfunction!(next, m)?)?;

    Ok(())
}
