use super::{Response, StepResponse};
use crate::{
    algorithms::online::uni_dimensional::randomized::{randomized, Memory},
    model::data_center::model::{
        DataCenterModel, DataCenterOfflineInput, DataCenterOnlineInput,
    },
    problem::FractionalSimplifiedSmoothedConvexOptimization,
    streaming::online,
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
) -> PyResult<Response<i32, Memory<'static>>> {
    let ((xs, cost), (int_xs, int_cost), m) = online::start(
        addr.parse().unwrap(),
        model,
        &randomized,
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
    addr: String,
    input: DataCenterOnlineInput,
) -> PyResult<StepResponse<i32, Memory<'static>>> {
    let ((x, cost), (int_x, int_cost), m) = online::next::<
        i32,
        FractionalSimplifiedSmoothedConvexOptimization,
        Memory,
        DataCenterOnlineInput,
    >(addr.parse().unwrap(), input);
    Ok(((x.to_vec(), cost), (int_x.to_vec(), int_cost), m))
}

/// Memoryless Algorithm
pub fn submodule(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(start, m)?)?;
    m.add_function(wrap_pyfunction!(next, m)?)?;

    Ok(())
}
