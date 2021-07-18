use crate::streaming::online;
use pyo3::prelude::*;

/// Stops backend server.
#[pyfunction]
fn stop(addr: &str) -> PyResult<()> {
    online::stop(addr.parse().unwrap());
    Ok(())
}

pub fn submodule(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(stop, m)?)?;

    Ok(())
}
