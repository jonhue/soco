use pyo3::prelude::*;
use crate::bindings::utils::make_package;

mod horizon_control;
mod lazy_budgeting;
mod online_balanced_descent;
mod online_gradient_descent;

pub fn submodule(py: Python, m: &PyModule) -> PyResult<()> {
    let horizon_control =
        PyModule::new(py, "horizon_control")?;
    horizon_control::submodule(py, horizon_control)?;
    make_package(
        py,
        horizon_control,
        "soco.data_center.online.multi_dimensional.horizon_control",
    );
    m.add_submodule(horizon_control)?;

    let lazy_budgeting =
        PyModule::new(py, "lazy_budgeting")?;
    lazy_budgeting::submodule(py, lazy_budgeting)?;
    make_package(
        py,
        lazy_budgeting,
        "soco.data_center.online.multi_dimensional.lazy_budgeting",
    );
    m.add_submodule(lazy_budgeting)?;

    let online_balanced_descent =
        PyModule::new(py, "online_balanced_descent")?;
    online_balanced_descent::submodule(py, online_balanced_descent)?;
    make_package(
        py,
        online_balanced_descent,
        "soco.data_center.online.multi_dimensional.online_balanced_descent",
    );
    m.add_submodule(online_balanced_descent)?;

    let online_gradient_descent =
        PyModule::new(py, "online_gradient_descent")?;
    online_gradient_descent::submodule(py, online_gradient_descent)?;
    make_package(
        py,
        online_gradient_descent,
        "soco.data_center.online.multi_dimensional.online_gradient_descent",
    );
    m.add_submodule(online_gradient_descent)?;

    Ok(())
}
