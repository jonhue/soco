use super::{
    DataCenterFractionalSimplifiedSmoothedConvexOptimization, Response,
    StepResponse,
};
use crate::{
    algorithms::online::uni_dimensional::memoryless::memoryless,
    model::data_center::{
        model::{
            DataCenterModel, DataCenterOfflineInput, DataCenterOnlineInput,
        },
        DataCenterModelOutputFailure, DataCenterModelOutputSuccess,
    },
    streaming::online::{self, OfflineResponse},
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
) -> PyResult<Response<f64, ()>> {
    let OfflineResponse {
        xs: (xs, cost),
        int_xs: (int_xs, int_cost),
        ..
    } = online::start(
        addr.parse().unwrap(),
        model,
        &memoryless,
        (),
        w,
        input,
        None,
    )
    .unwrap();
    Ok(((xs.to_vec(), cost), (int_xs.to_vec(), int_cost), None))
}

/// Executes next iteration of the algorithm.
#[pyfunction]
fn next(
    py: Python,
    addr: String,
    input: DataCenterOnlineInput,
) -> PyResult<StepResponse<f64, ()>> {
    py.allow_threads(|| {
        let ((x, cost), (int_x, int_cost), _) =
            online::next::<
                f64,
                DataCenterFractionalSimplifiedSmoothedConvexOptimization,
                (),
                DataCenterOnlineInput,
                DataCenterModelOutputSuccess,
                DataCenterModelOutputFailure,
            >(addr.parse().unwrap(), input)
            .map_err(PyAssertionError::new_err)?;
        Ok(((x.to_vec(), cost), (int_x.to_vec(), int_cost), None))
    })
}

/// Memoryless Algorithm
pub fn submodule(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(start, m)?)?;
    m.add_function(wrap_pyfunction!(next, m)?)?;

    Ok(())
}
