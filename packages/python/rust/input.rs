//! Validated, owned dense input for the Python numerical boundary.

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

/// A dense Gaussian problem whose storage is independent of Python objects.
#[pyclass(frozen, name = "_DenseInputs", module = "shrinkage._shrinkage")]
pub(crate) struct DenseInputs {
    shape: (usize, usize),
    columns: Vec<f64>,
    response: Vec<f64>,
}

impl DenseInputs {
    fn from_python(x: &Bound<'_, PyAny>, y: &Bound<'_, PyAny>) -> PyResult<Self> {
        let py = x.py();
        let numpy = py.import("numpy")?;
        let array_type = numpy.getattr("ndarray")?;
        let dtype = numpy.getattr("dtype")?.call1(("float64",))?;
        let x = array_buffer(x, "X", 2, &array_type, &dtype)?;
        let y = array_buffer(y, "y", 1, &array_type, &dtype)?;
        let shape = (x.shape()[0], x.shape()[1]);
        if shape.0 == 0 {
            return Err(PyValueError::new_err(
                "X must contain at least one observation",
            ));
        }
        if y.shape()[0] != shape.0 {
            return Err(PyValueError::new_err(format!(
                "y length ({}) must equal the number of X rows ({})",
                y.shape()[0],
                shape.0,
            )));
        }

        // Buffer exports retain the owners until both copies finish under the GIL.
        // No borrowed NumPy storage may reach a detached numerical operation.
        let data = Self {
            shape,
            columns: x.to_fortran_vec(py)?,
            response: y.to_vec(py)?,
        };
        drop((x, y));
        data.validate()?;
        Ok(data)
    }

    fn validate(&self) -> PyResult<()> {
        for (name, values) in [("X", &self.columns), ("y", &self.response)] {
            if values.iter().any(|value| !value.is_finite()) {
                return Err(PyValueError::new_err(format!(
                    "{name} must contain only finite values",
                )));
            }
        }
        Ok(())
    }
}

#[pymethods]
impl DenseInputs {
    #[getter]
    fn shape(&self) -> (usize, usize) {
        self.shape
    }

    /// Inspect the owned buffers while exercising detached native access.
    fn _snapshot(&self, py: Python<'_>) -> (Vec<f64>, Vec<f64>) {
        py.detach(|| (self.columns.clone(), self.response.clone()))
    }
}

fn array_buffer(
    value: &Bound<'_, PyAny>,
    name: &str,
    dimensions: usize,
    array_type: &Bound<'_, PyAny>,
    dtype: &Bound<'_, PyAny>,
) -> PyResult<PyBuffer<f64>> {
    // Exact arrays keep subclass hooks and implicit array protocols out of this boundary.
    if !value.get_type().is(array_type) {
        return Err(PyTypeError::new_err(format!(
            "{name} must be a numpy.ndarray; subclasses and array-like objects are unsupported",
        )));
    }
    if !value.getattr("dtype")?.eq(dtype)? {
        return Err(PyTypeError::new_err(format!(
            "{name} must have native-endian float64 dtype; convert explicitly with numpy.asarray",
        )));
    }
    if !value.getattr("flags")?.getattr("aligned")?.is_truthy()? {
        return Err(PyValueError::new_err(format!(
            "{name} must have aligned storage; make an aligned copy with numpy.require",
        )));
    }
    let actual_dimensions: usize = value.getattr("ndim")?.extract()?;
    if actual_dimensions != dimensions {
        let expected = if dimensions == 2 {
            "two-dimensional"
        } else {
            "one-dimensional"
        };
        return Err(PyValueError::new_err(format!(
            "{name} must be {expected}; got {} dimensions",
            actual_dimensions,
        )));
    }
    PyBuffer::<f64>::get(value)
}

/// Prepare the shared dense fitting boundary without exposing a fitting API yet.
#[pyfunction]
#[pyo3(signature = (x, y, *, copy = true))]
pub(crate) fn _prepare_dense_inputs(
    x: &Bound<'_, PyAny>,
    y: &Bound<'_, PyAny>,
    copy: bool,
) -> PyResult<DenseInputs> {
    if !copy {
        return Err(PyValueError::new_err(
            "copy=False is unsupported for NumPy inputs; numerical operations require Rust-owned storage",
        ));
    }
    DenseInputs::from_python(x, y)
}
