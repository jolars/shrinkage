//! Native support for the Shrinkage Python package.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use shrinkage::{L1, Penalty};

#[pyfunction]
fn _core_smoke_test() -> PyResult<f64> {
    L1::new(0.5)
        .value(&[2.0, -1.0])
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

#[pymodule]
fn _shrinkage(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add_function(wrap_pyfunction!(_core_smoke_test, module)?)?;
    Ok(())
}
