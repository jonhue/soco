use crate::{algorithms::online::multi_dimensional::lazy_budgeting::smoothed_balanced_load_optimization::{
        lb, Memory, Options,
    }, bindings::data_center::online::{
        Response,
        StepResponse,
    }, model::data_center::{
        model::{
            DataCenterModel, DataCenterOfflineInput, DataCenterOnlineInput,
        },
        DataCenterModelOutputFailure, DataCenterModelOutputSuccess,
    }, problem::IntegralSmoothedBalancedLoadOptimization, streaming::online::{self, OfflineResponse}};
use pyo3::{exceptions::PyAssertionError, prelude::*};

/// Starts backend in a new thread.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn start(
    py: Python,
    addr: String,
    model: DataCenterModel,
    input: DataCenterOfflineInput,
    w: i32,
    options: Options,
) -> PyResult<Response<i32, Memory>> {
    py.allow_threads(|| {
        let OfflineResponse {
            xs: (xs, cost),
            int_xs: (int_xs, int_cost),
            m,
            runtime,
        } = online::start(
            addr.parse().unwrap(),
            model,
            &lb,
            options,
            w,
            input,
            None,
        )
        .unwrap();
        Ok(((xs.to_vec(), cost), (int_xs.to_vec(), int_cost), m, runtime))
    })
}

/// Executes next iteration of the algorithm.
#[pyfunction]
fn next(
    py: Python,
    addr: String,
    input: DataCenterOnlineInput,
) -> PyResult<StepResponse<i32, Memory>> {
    py.allow_threads(|| {
        let ((x, cost), (int_x, int_cost), m, runtime) =
            online::next::<
                i32,
                IntegralSmoothedBalancedLoadOptimization,
                Memory,
                DataCenterOnlineInput,
                DataCenterModelOutputSuccess,
                DataCenterModelOutputFailure,
            >(addr.parse().unwrap(), input)
            .map_err(PyAssertionError::new_err)?;
        Ok(((x.to_vec(), cost), (int_x.to_vec(), int_cost), m, runtime))
    })
}

/// Lazy Capacity Provisioning
pub fn submodule(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(start, m)?)?;
    m.add_function(wrap_pyfunction!(next, m)?)?;

    m.add_class::<Options>()?;

    Ok(())
}
