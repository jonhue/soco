use pyo3::prelude::*;
use super::utils::make_package;

mod model;
mod offline;
mod online;

#[pymodule]
pub fn submodule(py: Python, m: &PyModule) -> PyResult<()> {
    let offline = PyModule::new(py, "offline")?;
    offline::submodule(py, offline)?;
    make_package(py, offline, "soco.data_center.offline");
    m.add_submodule(offline)?;

    let online = PyModule::new(py, "online")?;
    online::submodule(py, online)?;
    make_package(py, online, "soco.data_center.online");
    m.add_submodule(online)?;

    let model = PyModule::new(py, "model")?;
    model::submodule(py, model)?;
    make_package(py, model, "soco.data_center.model");
    m.add_submodule(model)?;

    Ok(())
}
