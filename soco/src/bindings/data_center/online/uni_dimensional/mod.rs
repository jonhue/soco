use crate::bindings::utils::make_package;
use pyo3::prelude::*;

mod lazy_capacity_provisioning;
mod memoryless;
mod probabilistic;
mod randomized;
mod randomly_biased_greedy;

pub fn submodule(py: Python, m: &PyModule) -> PyResult<()> {
    let lazy_capacity_provisioning =
        PyModule::new(py, "lazy_capacity_provisioning")?;
    lazy_capacity_provisioning::submodule(py, lazy_capacity_provisioning)?;
    make_package(
        py,
        lazy_capacity_provisioning,
        "soco.data_center.online.uni_dimensional.lazy_capacity_provisioning",
    );
    m.add_submodule(lazy_capacity_provisioning)?;

    let memoryless = PyModule::new(py, "memoryless")?;
    memoryless::submodule(py, memoryless)?;
    make_package(
        py,
        memoryless,
        "soco.data_center.online.uni_dimensional.memoryless",
    );
    m.add_submodule(memoryless)?;

    let probabilistic = PyModule::new(py, "probabilistic")?;
    probabilistic::submodule(py, probabilistic)?;
    make_package(
        py,
        probabilistic,
        "soco.data_center.online.uni_dimensional.probabilistic",
    );
    m.add_submodule(probabilistic)?;

    let randomized = PyModule::new(py, "randomized")?;
    randomized::submodule(py, randomized)?;
    make_package(
        py,
        randomized,
        "soco.data_center.online.uni_dimensional.randomized",
    );
    m.add_submodule(randomized)?;

    let randomly_biased_greedy = PyModule::new(py, "randomly_biased_greedy")?;
    randomly_biased_greedy::submodule(py, randomly_biased_greedy)?;
    make_package(
        py,
        randomly_biased_greedy,
        "soco.data_center.online.uni_dimensional.randomly_biased_greedy",
    );
    m.add_submodule(randomly_biased_greedy)?;

    Ok(())
}
