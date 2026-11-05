//! Preserve Rust path orchestration, complete penalties, and owned point fits.

use super::*;
use shrinkage::{PathFit, PathOptions, PathTermination, PenaltyMaximum, RuntimeDesign};

pub(super) type OptionsInput = (isize, Option<f64>, Option<f64>, bool, isize, f64, f64);
type Predictions = (usize, Vec<Vec<f64>>);

pub(super) enum Request {
    Explicit(Vec<Penalty>),
    Automatic(PathOptions),
}

impl Request {
    pub(super) fn parse(
        penalties: Option<Vec<PenaltyInput>>,
        options: Option<OptionsInput>,
    ) -> PyResult<Self> {
        match (penalties, options) {
            (Some(points), None) => Ok(Self::Explicit(
                points
                    .into_iter()
                    .map(Penalty::parse)
                    .collect::<PyResult<_>>()?,
            )),
            (None, Some((count, ratio, start, early, minimum, change, explained))) => {
                let mut options = PathOptions::new()
                    .nlambda(positive("nlambda", count)?)
                    .early_stopping(early)
                    .min_points(positive("min_points", minimum)?)
                    .deviance_change_tolerance(change)
                    .deviance_ratio_tolerance(explained);
                if let Some(ratio) = ratio {
                    options = options.lambda_min_ratio(ratio);
                }
                if let Some(start) = start {
                    options = options.lambda_start(start);
                }
                Ok(Self::Automatic(options))
            }
            _ => Err(PyValueError::new_err(
                "supply exactly one of penalties or options",
            )),
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn solve(
    method: Method,
    data: &DenseInputs,
    family: Family,
    fit_intercept: bool,
    normalization: Normalization,
    base: Penalty,
    request: Request,
) -> Result<PyModelPath, DetachedError> {
    let matrix = data.design.matrix();
    let nfeatures = matrix.ncols();
    let solver_name = match method {
        Method::Hybrid(_) => "hybrid",
        Method::Proximal(_) => "proximal_gradient",
    };
    // The same penalty type is required across points so hybrid workspaces can
    // retain their cluster storage, screening certificates, and working sets.
    macro_rules! solve {
        ($base:expr, $convert:expr, $record:expr) => {{
            let (points, options) = match request {
                Request::Explicit(points) => (
                    Some(
                        points
                            .into_iter()
                            .map($convert)
                            .collect::<Result<Vec<_>, _>>()?,
                    ),
                    None,
                ),
                Request::Automatic(options) => (None, Some(options)),
            };
            macro_rules! fit {
                ($problem:expr, $solver:expr, $wrap:expr) => {{
                    let problem = $problem.normalize(normalization);
                    let path = match points {
                        Some(points) => problem.fit_path_with(&$solver, points),
                        None => problem.fit_regularization_path_with(&$solver, &options.unwrap()),
                    }
                    .map_err(DetachedError::from)?;
                    Ok(collect(
                        path,
                        family,
                        solver_name,
                        nfeatures,
                        $wrap,
                        $record,
                    ))
                }};
            }
            macro_rules! family {
                ($datafit:expr) => {{
                    let datafit = $datafit.fit_intercept(fit_intercept);
                    match method {
                        Method::Hybrid(solver) => {
                            fit!(Problem::new(&matrix, datafit, $base), solver, |fit| {
                                NativeFit::Hybrid(Box::new(fit))
                            })
                        }
                        Method::Proximal(solver) => fit!(
                            Problem::new(
                                Box::new(MatrixDesign::<_, Array1<f64>>::new(&matrix))
                                    as Box<dyn RuntimeDesign>,
                                datafit,
                                $base,
                            ),
                            solver,
                            |fit| NativeFit::Proximal(Box::new(fit))
                        ),
                    }
                }};
            }
            match family {
                Family::Gaussian => family!(Gaussian::new(&data.response)),
                Family::Binomial => family!(Binomial::new(&data.response)),
            }
        }};
    }
    let mismatch = || {
        DetachedError::from(
            shrinkage::FitError::<std::convert::Infallible>::InvalidInput {
                message: "path penalties must have the same type as the problem penalty".into(),
            },
        )
    };
    match base {
        Penalty::Clustered(l1, fusion) => solve!(
            ClusteredLasso::new(l1, fusion),
            |penalty| match penalty {
                Penalty::Clustered(l1, fusion) => Ok(ClusteredLasso::new(l1, fusion)),
                _ => Err(mismatch()),
            },
            |p: &ClusteredLasso| ("clustered_lasso".into(), p.l1(), p.fusion(), None)
        ),
        Penalty::Chain(l1, fusion, weights) => solve!(
            chain(l1, fusion, weights),
            |penalty| match penalty {
                Penalty::Chain(l1, fusion, weights) => Ok(chain(l1, fusion, weights)),
                _ => Err(mismatch()),
            },
            |p: &ChainFusedLasso| (
                "chain_fused_lasso".into(),
                p.l1(),
                p.fusion(),
                p.weights().map(<[f64]>::to_vec)
            )
        ),
        Penalty::Slope(weights) => solve!(
            Slope::new(weights),
            |penalty| match penalty {
                Penalty::Slope(weights) => Ok(Slope::new(weights)),
                _ => Err(mismatch()),
            },
            |p: &Slope| ("slope".into(), 0.0, 0.0, Some(p.weights().to_vec()))
        ),
    }
}

fn collect<F, P>(
    path: PathFit<F, P>,
    family: Family,
    solver: &'static str,
    nfeatures: usize,
    wrap: impl Fn(F) -> NativeFit,
    record: impl Fn(&P) -> PenaltyInput,
) -> PyModelPath {
    let penalties = path.penalties().iter().map(record).collect();
    let lambdas = path.lambdas().map(<[f64]>::to_vec);
    let maximum_penalty = path.maximum_penalty().map(|maximum| match maximum {
        PenaltyMaximum::Finite(value) => value,
        PenaltyMaximum::Unbounded => f64::INFINITY,
    });
    let lambda_start = path.lambda_start();
    let termination = match path.termination() {
        PathTermination::Completed => "completed",
        PathTermination::DevianceChange => "deviance_change",
        PathTermination::DevianceRatio => "deviance_ratio",
        _ => "unknown",
    };
    let null_deviance = path.null_deviance();
    let deviances = path.deviances().map(<[f64]>::to_vec);
    let fits = path
        .into_parts()
        .0
        .into_iter()
        .map(|fit| PyModelFit {
            fit: Arc::new(wrap(fit)),
            family,
        })
        .collect();
    PyModelPath {
        fits,
        penalties,
        family,
        solver,
        nfeatures,
        lambdas,
        maximum_penalty,
        lambda_start,
        termination,
        null_deviance,
        deviances,
    }
}

#[pyclass(frozen, name = "_ModelPath", module = "shrinkage._shrinkage")]
pub(crate) struct PyModelPath {
    #[pyo3(get)]
    fits: Vec<PyModelFit>,
    #[pyo3(get)]
    penalties: Vec<PenaltyInput>,
    family: Family,
    #[pyo3(get)]
    solver: &'static str,
    #[pyo3(get)]
    nfeatures: usize,
    #[pyo3(get)]
    lambdas: Option<Vec<f64>>,
    #[pyo3(get)]
    maximum_penalty: Option<f64>,
    #[pyo3(get)]
    lambda_start: Option<f64>,
    #[pyo3(get)]
    termination: &'static str,
    #[pyo3(get)]
    null_deviance: Option<f64>,
    #[pyo3(get)]
    deviances: Option<Vec<f64>>,
}

#[pymethods]
impl PyModelPath {
    #[getter]
    fn npoints(&self) -> usize {
        self.fits.len()
    }
    #[getter]
    fn family(&self) -> &'static str {
        self.family.name()
    }
    #[getter]
    fn coefficients(&self) -> Vec<Vec<f64>> {
        self.fits.iter().map(PyModelFit::coefficients).collect()
    }
    #[getter]
    fn intercepts(&self) -> Vec<f64> {
        self.fits.iter().map(PyModelFit::intercept).collect()
    }
    #[pyo3(signature = (x, *, copy = true))]
    fn predict(&self, py: Python<'_>, x: &Bound<'_, PyAny>, copy: bool) -> PyResult<Predictions> {
        let data = DenseDesign::from_python(x, copy)?;
        let matrix = data.matrix();
        if matrix.ncols() != self.nfeatures {
            return Err(PyValueError::new_err(format!(
                "prediction matrix has {} columns; expected {}",
                matrix.ncols(),
                self.nfeatures
            )));
        }
        let nrows = matrix.nrows();
        let predictions = py
            .detach(|| {
                self.fits
                    .iter()
                    .map(|fit| match fit.fit.as_ref() {
                        NativeFit::Hybrid(fit) => fit.predict(&data.matrix()),
                        NativeFit::Proximal(fit) => fit.predict(&data.matrix()),
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .map_err(fit_error)?;
        Ok((nrows, predictions))
    }
}
