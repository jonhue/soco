use pyo3::prelude::*;

mod model;
mod offline;
mod online;

#[pymodule]
pub fn submodule(py: Python, _m: &PyModule) -> PyResult<()> {
    let offline = PyModule::new(py, "offline")?;
    offline::submodule(py, offline)?;

    let online = PyModule::new(py, "online")?;
    online::submodule(py, online)?;

    let model = PyModule::new(py, "model")?;
    model::submodule(py, model)?;

    Ok(())
}
