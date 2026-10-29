//! Runtime composition without instantiating every combination of components.

use std::{error::Error, marker::PhantomData};

use lazymatrix::{ColumnStats, MatTransposeVecInto, MatVecInto, RawColumns};

use crate::error::{invalid, unsupported};
use crate::problem::{prepare_normalization, validate_matrix};
use crate::screening::ScreeningDesign;
use crate::solver::proximal::{NativeDesign, iterate, iterate_path};
use crate::{
    BackendError, CoordinateDescent, FitError, LassoFit, Normalization, Preprocessing, Problem,
    ProximalDesign, ProximalFit, ProximalGradient, ProximalPenalty, ProximalVector, SmoothDatafit,
    Solver,
};

/// A problem whose design, smooth datafit, and complete proximal term are
/// selected independently before fitting.
///
/// The shared iteration calls object-safe oracles for complete products, loss
/// and gradient evaluations, and proximal operations. It does not erase an
/// already composed typed problem. Use [`MatrixDesign`] to adapt each concrete
/// backend once, independently of the chosen datafit and penalty.
///
/// [`ProximalGradient<Vec<f64>>`] fits this problem, while adapters retain each
/// backend's native vector type. Normalization and supplied vectors use the
/// same builders and semantics as the typed [`Problem`].
/// Selecting [`CoordinateDescent`] returns [`FitError::UnsupportedCombination`]
/// before preparing the design: these oracles do not expose its Gaussian/L1
/// coordinate updates and dual certificate. Use proximal gradient here, or a
/// typed Gaussian/L1 problem with coordinate descent. A value-only datafit or
/// penalty cannot be boxed into the required capability traits.
///
/// ```compile_fail
/// use shrinkage::{FitError, Penalty, ProximalPenalty};
/// fn select_penalty<P: Penalty<Error = FitError> + 'static>(penalty: P)
///     -> Box<dyn ProximalPenalty>
/// {
///     Box::new(penalty)
/// }
/// ```
///
/// ```compile_fail
/// use shrinkage::{Datafit, FitError, SmoothDatafit};
/// fn select_datafit<D: Datafit<Error = FitError> + 'static>(datafit: D)
///     -> Box<dyn SmoothDatafit>
/// {
///     Box::new(datafit)
/// }
/// ```
///
/// ```
/// # #[cfg(feature = "faer_v0_24")] {
/// use faer::{Col, Mat};
/// use shrinkage::{Gaussian, MatrixDesign, ProximalGradient, Ridge, RuntimeProblem};
/// let x = Mat::from_fn(3, 1, |i, _| i as f64);
/// let y = [1.0, 3.0, 5.0];
/// let problem = RuntimeProblem::new(
///     Box::new(MatrixDesign::<_, Col<f64>>::new(&x)),
///     Box::new(Gaussian::new(&y)),
///     Box::new(Ridge::new(0.1)),
/// );
/// let fit = problem.fit_with(&ProximalGradient::new())?;
/// let predictions = fit.predict(&x)?;
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub type RuntimeProblem<'a> = Problem<
    Box<dyn RuntimeDesign + 'a>,
    Box<dyn SmoothDatafit + 'a>,
    Box<dyn ProximalPenalty + 'a>,
>;

/// Runtime composition retaining the capabilities needed for automatic paths.
///
/// Box a [`crate::PathDatafit`] and an [`crate::ErasedPathPenalty`] before
/// erasure. Explicit paths still support the broader [`RuntimeProblem`].
pub type RuntimePathProblem<'a> = Problem<
    Box<dyn RuntimeDesign + 'a>,
    Box<dyn crate::PathDatafit + 'a>,
    Box<dyn crate::PathPenalty<AtStrength = Box<dyn ProximalPenalty>> + 'a>,
>;

/// Multivariate Gaussian fitting with a runtime-selected design and penalty.
///
/// [`crate::AxisPenalty`] specifies the coefficient layout and penalty axis.
/// Each response has its own unpenalized intercept. The shared proximal solver
/// fits all responses jointly, including penalties that couple them.
pub type RuntimeMultivariateProblem<'a> = Problem<
    Box<dyn RuntimeDesign + 'a>,
    crate::MultivariateGaussian<'a>,
    crate::AxisPenalty<Box<dyn ProximalPenalty + 'a>>,
>;

/// Object-safe preparation of a runtime-selected design.
///
/// Entry validation and generic column views remain in the concrete adapter.
/// Preparation allocates reusable native product buffers once per fit and
/// resolves automatic normalization using the selected datafit's intercept.
pub trait RuntimeDesign {
    /// Number of training observations.
    fn nrows(&self) -> usize;

    /// Number of training features.
    fn ncols(&self) -> usize;

    /// Validate the design and prepare its normalized product oracle.
    ///
    /// # Errors
    /// Reject invalid data or normalization vectors and preserve failures
    /// from backend statistics with their operation context and source.
    fn prepare(
        &self,
        normalization: Normalization,
        centers: Option<&[f64]>,
        scales: Option<&[f64]>,
        fit_intercept: bool,
    ) -> Result<Box<dyn ProximalDesign<Error = BackendError> + '_>, FitError<BackendError>>;
}

/// A borrowed concrete LazyMatrix backend for runtime proximal composition.
///
/// `V` is the native vector accepted by the backend's products. This adapter
/// retains static column access and never depends on a datafit or penalty type.
pub struct MatrixDesign<'a, M: ?Sized, V = Vec<f64>> {
    matrix: &'a M,
    vector: PhantomData<V>,
}

impl<'a, M: ?Sized, V> MatrixDesign<'a, M, V> {
    /// Borrow a backend; validation and normalization occur during fitting.
    pub fn new(matrix: &'a M) -> Self {
        Self {
            matrix,
            vector: PhantomData,
        }
    }
}

impl<M, V> RuntimeDesign for MatrixDesign<'_, M, V>
where
    M: RawColumns<f64> + ColumnStats<f64> + MatVecInto<V> + MatTransposeVecInto<V> + ?Sized,
    M::Error: Error + 'static,
    V: ProximalVector + 'static,
{
    fn nrows(&self) -> usize {
        self.matrix.nrows()
    }

    fn ncols(&self) -> usize {
        self.matrix.ncols()
    }

    fn prepare(
        &self,
        normalization: Normalization,
        centers: Option<&[f64]>,
        scales: Option<&[f64]>,
        fit_intercept: bool,
    ) -> Result<Box<dyn ProximalDesign<Error = BackendError> + '_>, FitError<BackendError>> {
        if self.nrows() == 0 {
            return Err(invalid("training data must have at least one observation"));
        }
        validate_matrix::<_, M::Error>(self.matrix).map_err(FitError::erase_backend)?;
        let prepared =
            prepare_normalization(self.matrix, normalization, centers, scales, fit_intercept)
                .map_err(FitError::erase_backend)?;
        Ok(Box::new(ErasedDesign(
            NativeDesign::<_, V>::new_screenable(prepared),
        )))
    }
}

struct ErasedDesign<M, V>(NativeDesign<M, V>);

impl<M, V> ProximalDesign for ErasedDesign<M, V>
where
    M: MatVecInto<V> + MatTransposeVecInto<V>,
    M::Error: Error + 'static,
    V: ProximalVector,
{
    type Error = BackendError;

    fn nrows(&self) -> usize {
        self.0.nrows()
    }

    fn ncols(&self) -> usize {
        self.0.ncols()
    }

    fn forward(&mut self, input: &[f64], output: &mut [f64]) -> Result<(), FitError<BackendError>> {
        self.0
            .forward(input, output)
            .map_err(FitError::erase_backend)
    }

    fn transpose(
        &mut self,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), FitError<BackendError>> {
        self.0
            .transpose(input, output)
            .map_err(FitError::erase_backend)
    }

    fn into_preprocessing(self: Box<Self>) -> Preprocessing {
        self.0.into_preprocessing()
    }

    fn screening_design(&mut self) -> Option<&mut dyn ScreeningDesign<Error = Self::Error>> {
        self.0.screening_design()?;
        Some(self)
    }
}

impl<M, V> ScreeningDesign for ErasedDesign<M, V>
where
    M: MatVecInto<V> + MatTransposeVecInto<V>,
    M::Error: Error + 'static,
    V: ProximalVector,
{
    type Error = BackendError;

    fn column_norms(&mut self, output: &mut [f64]) -> Result<(), FitError<Self::Error>> {
        self.0.column_norms(output).map_err(FitError::erase_backend)
    }

    fn column_error_scales(&mut self, output: &mut [f64]) -> Result<(), FitError<Self::Error>> {
        self.0
            .column_error_scales(output)
            .map_err(FitError::erase_backend)
    }

    fn forward_active(
        &mut self,
        input: &[f64],
        active: &[usize],
        output: &mut [f64],
    ) -> Result<(), FitError<Self::Error>> {
        self.0
            .forward_active(input, active, output)
            .map_err(FitError::erase_backend)
    }

    fn transpose_active(
        &mut self,
        input: &[f64],
        active: &[usize],
        output: &mut [f64],
    ) -> Result<(), FitError<Self::Error>> {
        self.0
            .transpose_active(input, active, output)
            .map_err(FitError::erase_backend)
    }
}

impl ProximalGradient<Vec<f64>> {
    /// Fit runtime-selected penalties in order with one prepared design.
    ///
    /// Reuses native products and iteration storage and warms each point from
    /// the preceding optimization-scale parameters. The problem's penalty is
    /// replaced by the path entries. Steps and iteration budgets reset at each
    /// point; iteration-limit fits are retained as warm starts.
    ///
    /// # Errors
    /// Returns validation or fitting errors without a partial path.
    pub fn solve_runtime_path(
        &self,
        problem: &RuntimeProblem<'_>,
        penalties: &[Box<dyn ProximalPenalty + '_>],
    ) -> Result<Vec<ProximalFit>, FitError<BackendError>> {
        let threshold = self.validate()?;
        if problem.datafit().nobs() != problem.design().nrows() {
            return Err(invalid(
                "datafit observation count must match training rows",
            ));
        }
        if penalties.is_empty() {
            return Ok(Vec::new());
        }
        let mut design = problem.design().prepare(
            problem.normalization(),
            problem.centers(),
            problem.scales(),
            problem.datafit().fits_intercept(),
        )?;
        let solutions = iterate_path(
            design.as_mut(),
            problem.datafit().as_ref(),
            penalties,
            self,
            threshold,
        )?;
        let preprocessing = design.into_preprocessing();
        solutions
            .into_iter()
            .map(|solution| ProximalFit::from_solution(solution, preprocessing.clone()))
            .collect()
    }
}

impl<D: SmoothDatafit, P: ProximalPenalty> Solver<Problem<Box<dyn RuntimeDesign + '_>, D, P>>
    for ProximalGradient<Vec<f64>>
{
    type Fit = ProximalFit;
    type Error = FitError<BackendError>;

    fn solve(
        &self,
        problem: &Problem<Box<dyn RuntimeDesign + '_>, D, P>,
    ) -> Result<Self::Fit, Self::Error> {
        let threshold = self.validate()?;
        if problem.datafit().nobs() != problem.design().nrows() {
            return Err(invalid(
                "datafit observation count must match training rows",
            ));
        }
        let mut design = problem.design().prepare(
            problem.normalization(),
            problem.centers(),
            problem.scales(),
            problem.datafit().fits_intercept(),
        )?;
        let solution = iterate(
            design.as_mut(),
            problem.datafit(),
            problem.penalty(),
            self,
            threshold,
        )?;
        ProximalFit::from_solution(solution, design.into_preprocessing())
    }
}

impl Solver<RuntimeProblem<'_>> for CoordinateDescent {
    type Fit = LassoFit;
    type Error = FitError<BackendError>;

    fn solve(&self, _problem: &RuntimeProblem<'_>) -> Result<Self::Fit, Self::Error> {
        Err(unsupported(
            "CoordinateDescent",
            "runtime proximal oracles do not expose Gaussian/L1 coordinate updates or a dual certificate",
            "use ProximalGradient<Vec<f64>> for RuntimeProblem, or a typed Problem with Gaussian and L1 for CoordinateDescent",
        ))
    }
}
