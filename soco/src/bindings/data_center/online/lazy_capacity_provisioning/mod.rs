use pyo3::prelude::*;

mod fractional;
mod integral;

pub fn submodule(py: Python, m: &PyModule) -> PyResult<()> {
    let fractional = PyModule::new(py, "fractional")?;
    fractional::submodule(py, fractional)?;
    m.add_submodule(fractional)?;

    let integral = PyModule::new(py, "integral")?;
    integral::submodule(py, integral)?;
    m.add_submodule(fractional)?;

    Ok(())
}
