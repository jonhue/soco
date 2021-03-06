use crate::bindings::data_center::online::{
    DataCenterFractionalSimplifiedSmoothedConvexOptimization, Response,
    StepResponse,
};
use crate::{
    algorithms::online::uni_dimensional::probabilistic::{
        probabilistic, Memory, Options,
    },
    breakpoints::Breakpoints,
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
    py: Python,
    addr: String,
    model: DataCenterModel,
    input: DataCenterOfflineInput,
    w: i32,
    options: Options,
) -> PyResult<Response<f64, Memory<'static>>> {
    py.allow_threads(|| {
        let OfflineResponse {
            xs: (xs, cost),
            int_xs: (int_xs, int_cost),
            m,
            runtime,
        } = online::start(
            addr.parse().unwrap(),
            model,
            &probabilistic,
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
) -> PyResult<StepResponse<f64, Memory<'static>>> {
    py.allow_threads(|| {
        let ((x, cost), (int_x, int_cost), m, runtime) =
            online::next::<
                f64,
                DataCenterFractionalSimplifiedSmoothedConvexOptimization,
                Memory,
                DataCenterOnlineInput,
                DataCenterModelOutputSuccess,
                DataCenterModelOutputFailure,
            >(addr.parse().unwrap(), input)
            .map_err(PyAssertionError::new_err)?;
        Ok(((x.to_vec(), cost), (int_x.to_vec(), int_cost), m, runtime))
    })
}

/// Memoryless Algorithm
pub fn submodule(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(start, m)?)?;
    m.add_function(wrap_pyfunction!(next, m)?)?;

    m.add_class::<Options>()?;
    m.add_class::<Breakpoints>()?;

    Ok(())
}
