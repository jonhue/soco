use super::{Response, StepResponse};
use crate::{
    algorithms::online::uni_dimensional::probabilistic::{
        probabilistic, Memory, Options,
    },
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
    options: Options,
) -> PyResult<Response<f64, Memory<'static>>> {
    let (xs, cost, _) = online::start(
        addr.parse().unwrap(),
        model,
        &probabilistic,
        options,
        w,
        input,
        None,
    )
    .unwrap();
    Ok((xs.to_vec(), cost, None))
}

/// Executes next iteration of the algorithm.
#[pyfunction]
fn next(
    addr: String,
    input: DataCenterOnlineInput,
) -> PyResult<StepResponse<f64, Memory<'static>>> {
    let (x, cost, _) = online::next::<
        f64,
        FractionalSimplifiedSmoothedConvexOptimization,
        Memory,
        DataCenterOnlineInput,
    >(addr.parse().unwrap(), input);
    Ok((x.to_vec(), cost, None))
}

/// Memoryless Algorithm
pub fn submodule(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(start, m)?)?;
    m.add_function(wrap_pyfunction!(next, m)?)?;

    m.add_class::<Options>()?;

    Ok(())
}
