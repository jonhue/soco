use crate::{
    algorithms::offline::uni_dimensional::capacity_provisioning::brcp,
    model::data_center::model::{DataCenterModel, DataCenterOfflineInput},
    streaming::offline,
};
use pyo3::prelude::*;

/// Stops backend server.
#[pyfunction]
fn brcp_py(
    model: DataCenterModel,
    input: DataCenterOfflineInput,
    inverted: bool,
) -> PyResult<Vec<Vec<f64>>> {
    Ok(offline::solve(&model, &brcp, (), input, inverted)
        .unwrap()
        .to_vec())
}

pub fn submodule(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(brcp_py, m)?)?;

    Ok(())
}
