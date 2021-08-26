use pyo3::prelude::*;

mod dual;
mod greedy;
mod meta;
mod primal;
mod regularized;

pub fn submodule(py: Python, m: &PyModule) -> PyResult<()> {
    let dual = PyModule::new(py, "dual")?;
    dual::submodule(py, dual)?;
    m.add_submodule(dual)?;

    let greedy = PyModule::new(py, "greedy")?;
    greedy::submodule(py, greedy)?;
    m.add_submodule(greedy)?;

    let meta = PyModule::new(py, "meta")?;
    meta::submodule(py, meta)?;
    m.add_submodule(meta)?;

    let primal = PyModule::new(py, "primal")?;
    primal::submodule(py, primal)?;
    m.add_submodule(primal)?;

    let regularized = PyModule::new(py, "regularized")?;
    regularized::submodule(py, regularized)?;
    m.add_submodule(regularized)?;

    Ok(())
}
