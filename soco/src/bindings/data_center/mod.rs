use pyo3::prelude::*;

mod model;
mod offline;
mod online;

#[pymodule]
pub fn submodule(py: Python, m: &PyModule) -> PyResult<()> {
    let offline = PyModule::new(py, "offline")?;
    offline::submodule(py, offline)?;
    m.add_submodule(offline)?;

    let online = PyModule::new(py, "online")?;
    online::submodule(py, online)?;
    m.add_submodule(online)?;

    let model = PyModule::new(py, "model")?;
    model::submodule(py, model)?;
    m.add_submodule(model)?;

    Ok(())
}
