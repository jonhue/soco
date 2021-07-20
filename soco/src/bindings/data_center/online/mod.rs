use crate::streaming::online;
use pyo3::prelude::*;

mod lazy_capacity_provisioning;
mod memoryless;
mod probabilistic;
mod randomized;
mod randomly_biased_greedy;

type Response<T, M> = (Vec<Vec<T>>, f64, Option<M>);
type StepResponse<T, M> = (Vec<T>, f64, Option<M>);

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

    let memoryless = PyModule::new(py, "memoryless")?;
    memoryless::submodule(py, memoryless)?;

    let probabilistic = PyModule::new(py, "probabilistic")?;
    probabilistic::submodule(py, probabilistic)?;

    let randomized = PyModule::new(py, "randomized")?;
    randomized::submodule(py, randomized)?;

    let randomly_biased_greedy = PyModule::new(py, "randomly_biased_greedy")?;
    randomly_biased_greedy::submodule(py, randomly_biased_greedy)?;

    Ok(())
}
