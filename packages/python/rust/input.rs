//! Validated, owned dense input for the Python numerical boundary.

use ndarray::{Array2, ShapeBuilder};
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

/// A column-major snapshot with no borrowed Python storage.
pub(crate) struct DenseDesign {
    pub(crate) matrix: Array2<f64>,
}

impl DenseDesign {
    pub(crate) fn from_python(x: &Bound<'_, PyAny>, copy: bool) -> PyResult<Self> {
        require_copy(copy)?;
        let numpy = x.py().import("numpy")?;
        let array_type = numpy.getattr("ndarray")?;
        let dtype = numpy.getattr("dtype")?.call1(("float64",))?;
        let buffer = array_buffer(x, "X", 2, &array_type, &dtype)?;
        Self::from_buffer(&buffer, x.py())
    }

    fn from_buffer(buffer: &PyBuffer<f64>, py: Python<'_>) -> PyResult<Self> {
        let shape = (buffer.shape()[0], buffer.shape()[1]);
        let columns = buffer.to_fortran_vec(py)?;
        validate_finite("X", &columns)?;
        let matrix = Array2::from_shape_vec(shape.f(), columns)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self { matrix })
    }
}

/// A dense single-response problem whose storage is independent of Python objects.
#[pyclass(frozen, name = "_DenseInputs", module = "shrinkage._shrinkage")]
pub(crate) struct DenseInputs {
    pub(crate) design: DenseDesign,
    pub(crate) response: Vec<f64>,
}

impl DenseInputs {
    pub(crate) fn from_python(
        x: &Bound<'_, PyAny>,
        y: &Bound<'_, PyAny>,
        copy: bool,
    ) -> PyResult<Self> {
        require_copy(copy)?;
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
            design: DenseDesign::from_buffer(&x, py)?,
            response: y.to_vec(py)?,
        };
        drop((x, y));
        validate_finite("y", &data.response)?;
        Ok(data)
    }
}

#[pymethods]
impl DenseInputs {
    #[getter]
    fn shape(&self) -> (usize, usize) {
        self.design.matrix.dim()
    }

    /// Inspect the owned buffers while exercising detached native access.
    fn _snapshot(&self, py: Python<'_>) -> (Vec<f64>, Vec<f64>) {
        py.detach(|| {
            (
                self.design.matrix.as_slice_memory_order().unwrap().to_vec(),
                self.response.clone(),
            )
        })
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

fn require_copy(copy: bool) -> PyResult<()> {
    if !copy {
        return Err(PyValueError::new_err(
            "copy=False is unsupported for NumPy inputs; numerical operations require Rust-owned storage",
        ));
    }
    Ok(())
}

fn validate_finite(name: &str, values: &[f64]) -> PyResult<()> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err(format!(
            "{name} must contain only finite values",
        )));
    }
    Ok(())
}

/// Exercise the shared dense fitting boundary independently of a solver.
#[pyfunction]
#[pyo3(signature = (x, y, *, copy = true))]
pub(crate) fn _prepare_dense_inputs(
    x: &Bound<'_, PyAny>,
    y: &Bound<'_, PyAny>,
    copy: bool,
) -> PyResult<DenseInputs> {
    DenseInputs::from_python(x, y, copy)
}
