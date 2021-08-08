use pyo3::prelude::*;

mod smoothed_balanced_load_optimization;
mod smoothed_load_optimization;

pub fn submodule(py: Python, m: &PyModule) -> PyResult<()> {
    let smoothed_balanced_load_optimization = PyModule::new(py, "smoothed_balanced_load_optimization")?;
    smoothed_balanced_load_optimization::submodule(py, smoothed_balanced_load_optimization)?;
    m.add_submodule(smoothed_balanced_load_optimization)?;

    let smoothed_load_optimization = PyModule::new(py, "smoothed_load_optimization")?;
    smoothed_load_optimization::submodule(py, smoothed_load_optimization)?;
    m.add_submodule(smoothed_load_optimization)?;

    Ok(())
}
