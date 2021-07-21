use pyo3::prelude::*;
use utils::make_package;

mod data_center;
mod utils;

#[pymodule]
pub fn soco(py: Python, m: &PyModule) -> PyResult<()> {
    let data_center = PyModule::new(py, "data_center")?;
    data_center::submodule(py, data_center)?;
    make_package(py, data_center, "soco.data_center");
    m.add_submodule(data_center)?;

    Ok(())
}
