//! Gaussian lasso calls over owned input, detached from the Python interpreter.

use pyo3::prelude::*;
use shrinkage::{Lasso, LassoFit};

use crate::common::{
    self, fit_error, normalization as normalization_name, parse_stopping, positive,
};

use crate::input::{DenseDesign, DenseInputs};

#[pyclass(frozen, name = "_Lasso", module = "shrinkage._shrinkage")]
pub(crate) struct PyLasso {
    model: Lasso,
}

#[pymethods]
impl PyLasso {
    #[new]
    fn new(
        lambda_: f64,
        fit_intercept: bool,
        normalization: &str,
        stopping: Option<(String, f64, f64)>,
        max_iterations: isize,
    ) -> PyResult<Self> {
        let normalization = normalization_name(normalization)?;
        let max_iterations = positive("max_iterations", max_iterations)?;
        let mut model = Lasso::new(lambda_)
            .fit_intercept(fit_intercept)
            .normalize(normalization)
            .max_iterations(max_iterations);
        if let Some(stopping) = stopping {
            model = model.terminate_on(parse_stopping(stopping)?);
        }
        Ok(Self { model })
    }

    #[pyo3(signature = (x, y, *, copy = true))]
    fn fit(
        &self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        y: &Bound<'_, PyAny>,
        copy: bool,
    ) -> PyResult<PyLassoFit> {
        let data = DenseInputs::from_python(x, y, copy)?;
        // All buffer exports have ended, and the closure reads only Rust-owned data.
        let fit = py.detach(|| self.model.fit(&data.design.matrix, &data.response));
        fit.map(|fit| PyLassoFit { fit }).map_err(fit_error)
    }
}

#[pyclass(frozen, name = "_LassoFit", module = "shrinkage._shrinkage")]
pub(crate) struct PyLassoFit {
    fit: LassoFit,
}

#[pymethods]
impl PyLassoFit {
    #[getter]
    fn coefficients(&self) -> Vec<f64> {
        self.fit.coefficients().to_vec()
    }
    #[getter]
    fn intercept(&self) -> f64 {
        self.fit.intercept()
    }
    #[getter]
    fn centers(&self) -> Option<Vec<f64>> {
        self.fit.preprocessing().centers().map(<[f64]>::to_vec)
    }
    #[getter]
    fn scales(&self) -> Option<Vec<f64>> {
        self.fit.preprocessing().scales().map(<[f64]>::to_vec)
    }
    #[getter]
    fn centering(&self) -> &'static str {
        common::centering(self.fit.preprocessing().centering())
    }
    #[getter]
    fn scaling(&self) -> &'static str {
        common::scaling(self.fit.preprocessing().scaling())
    }
    #[getter]
    fn termination(&self) -> &'static str {
        common::termination(self.fit.termination())
    }
    #[getter]
    fn iterations(&self) -> usize {
        self.fit.iterations()
    }
    #[getter]
    fn objective(&self) -> f64 {
        self.fit.objective()
    }
    #[getter]
    fn kkt_violation(&self) -> f64 {
        self.fit.kkt_violation()
    }
    #[getter]
    fn dual_certificate(&self) -> Vec<f64> {
        self.fit.dual_certificate().to_vec()
    }
    #[getter]
    fn dual_objective(&self) -> f64 {
        self.fit.dual_objective()
    }
    #[getter]
    fn duality_gap(&self) -> f64 {
        self.fit.duality_gap()
    }
    #[getter]
    fn reference_loss(&self) -> f64 {
        self.fit.reference_loss()
    }
    #[getter]
    fn stopping(&self) -> (&'static str, f64, f64) {
        common::stopping(self.fit.stopping_criterion())
    }
    #[getter]
    fn stopping_value(&self) -> f64 {
        self.fit.stopping_value()
    }
    #[getter]
    fn stopping_threshold(&self) -> f64 {
        self.fit.stopping_threshold()
    }

    #[pyo3(signature = (x, *, copy = true))]
    fn predict(&self, py: Python<'_>, x: &Bound<'_, PyAny>, copy: bool) -> PyResult<Vec<f64>> {
        let data = DenseDesign::from_python(x, copy)?;
        py.detach(|| self.fit.predict(&data.matrix))
            .map_err(fit_error)
    }
}
