use pyo3::prelude::*;

mod probabilistic;
mod randomly_biased_greedy;

pub fn submodule(py: Python, m: &PyModule) -> PyResult<()> {
    let probabilistic = PyModule::new(py, "probabilistic")?;
    probabilistic::submodule(py, probabilistic)?;
    m.add_submodule(probabilistic)?;

    let randomly_biased_greedy = PyModule::new(py, "randomly_biased_greedy")?;
    randomly_biased_greedy::submodule(py, randomly_biased_greedy)?;
    m.add_submodule(randomly_biased_greedy)?;

    Ok(())
}
