use pyo3::{prelude::*, py_run};

pub fn make_package(py: Python, module: &PyModule, path: &str) {
    py_run!(
        py,
        module,
        &format!("import sys; sys.modules['{}'] = module", path)[..]
    );
}
