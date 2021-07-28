use crate::{
    algorithms::online::uni_dimensional::lazy_capacity_provisioning::{
        lcp, Memory,
    },
    bindings::data_center::online::{Response, StepResponse},
    model::data_center::model::{
        DataCenterModel, DataCenterOfflineInput, DataCenterOnlineInput,
    },
    problem::IntegralSimplifiedSmoothedConvexOptimization,
    streaming::online,
};
use pyo3::{exceptions::PyAssertionError, prelude::*};

/// Starts backend in a new thread.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn start(
    addr: String,
    model: DataCenterModel,
    input: DataCenterOfflineInput,
    w: i32,
) -> PyResult<Response<i32, Memory<i32>>> {
    let ((xs, cost), (int_xs, int_cost), m) = online::start(
        addr.parse().unwrap(),
        model,
        &lcp::<i32, IntegralSimplifiedSmoothedConvexOptimization>,
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
) -> PyResult<StepResponse<i32, Memory<i32>>> {
    py.allow_threads(|| {
        let ((x, cost), (int_x, int_cost), m) =
            online::next::<
                i32,
                IntegralSimplifiedSmoothedConvexOptimization,
                Memory<i32>,
                DataCenterOnlineInput,
            >(addr.parse().unwrap(), input)
            .map_err(PyAssertionError::new_err)?;
        Ok(((x.to_vec(), cost), (int_x.to_vec(), int_cost), m))
    })
}

/// Lazy Capacity Provisioning
pub fn submodule(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(start, m)?)?;
    m.add_function(wrap_pyfunction!(next, m)?)?;

    Ok(())
}
