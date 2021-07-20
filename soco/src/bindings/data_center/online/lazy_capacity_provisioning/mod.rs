use pyo3::prelude::*;

mod fractional;
mod integral;

pub fn submodule(py: Python, _m: &PyModule) -> PyResult<()> {
    let fractional = PyModule::new(py, "fractional")?;
    fractional::submodule(py, fractional)?;

    let integral = PyModule::new(py, "integral")?;
    integral::submodule(py, integral)?;

    Ok(())
}
