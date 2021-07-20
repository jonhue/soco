use pyo3::prelude::*;

mod data_center;

#[pymodule]
pub fn supermodule(py: Python, _m: &PyModule) -> PyResult<()> {
    let data_center = PyModule::new(py, "data_center")?;
    data_center::submodule(py, data_center)?;

    Ok(())
}
