use pyo3::prelude::*;

mod averaging_fixed_horizon_control;
mod receding_horizon_control;

pub fn submodule(py: Python, m: &PyModule) -> PyResult<()> {
    let averaging_fixed_horizon_control = PyModule::new(py, "averaging_fixed_horizon_control")?;
    averaging_fixed_horizon_control::submodule(py, averaging_fixed_horizon_control)?;
    m.add_submodule(averaging_fixed_horizon_control)?;

    let receding_horizon_control = PyModule::new(py, "receding_horizon_control")?;
    receding_horizon_control::submodule(py, receding_horizon_control)?;
    m.add_submodule(receding_horizon_control)?;

    Ok(())
}
