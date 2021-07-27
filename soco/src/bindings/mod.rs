use pyo3::prelude::*;
use utils::make_package;

use crate::{
    cost::Cost,
    model::data_center::{
        DataCenterModelOutputFailure, DataCenterModelOutputSuccess,
    },
};

mod data_center;
mod utils;

type DataCenterCost =
    Cost<DataCenterModelOutputSuccess, DataCenterModelOutputFailure>;

#[pymodule]
pub fn soco(py: Python, m: &PyModule) -> PyResult<()> {
    pyo3_log::init();

    let data_center = PyModule::new(py, "data_center")?;
    data_center::submodule(py, data_center)?;
    make_package(py, data_center, "soco.data_center");
    m.add_submodule(data_center)?;

    Ok(())
}
