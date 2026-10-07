//! Composition and validated preparation of training problems.

use crate::error::{finite, invalid};
use crate::solver::{Solver, coordinate};
use crate::{Centering, LassoError, LassoFit, Normalization, Preprocessing, Scaling};
pub use crate::{datafit::Gaussian, penalty::L1, solver::CoordinateDescent};
use lazymatrix::{ColumnStats, LazyMatrix, RawColumn, RawColumns};

/// A typed design, datafit, and penalty with training normalization settings.
///
/// [`CoordinateDescent`] supports a borrowed [`RawColumns`] design with
/// [`Gaussian`] and [`L1`], returning [`LassoFit`]. [`crate::ProximalGradient`]
/// uses native forward and transposed products with [`crate::SmoothDatafit`]
/// and a complete [`crate::ProximalPenalty`], returning [`crate::ProximalFit`].
/// Both results retain original-scale parameters and training normalization.
///
/// ```
/// # #[cfg(feature = "faer_v0_24")] {
/// use faer::Mat;
/// use shrinkage::{CoordinateDescent, Gaussian, L1, Normalization, Problem};
///
/// let x = Mat::from_fn(3, 1, |i, _| i as f64);
/// let y = [1.0, 3.0, 5.0];
/// let fit = Problem::new(&x, Gaussian::new(&y), L1::new(0.1))
///     .normalize(Normalization::Standardize)
///     .fit_with(&CoordinateDescent::new())?;
/// let predictions = fit.predict(&x)?;
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Debug)]
pub struct Problem<X, D, P> {
    design: X,
    datafit: D,
    penalty: P,
    normalization: Normalization,
    supplied_centers: Option<Vec<f64>>,
    supplied_scales: Option<Vec<f64>>,
}

impl<X, D, P> Problem<X, D, P> {
    /// Compose a design, datafit, and penalty.
    pub fn new(design: X, datafit: D, penalty: P) -> Self {
        Self {
            design,
            datafit,
            penalty,
            normalization: Normalization::Auto,
            supplied_centers: None,
            supplied_scales: None,
        }
    }

    /// Borrow the design component.
    pub fn design(&self) -> &X {
        &self.design
    }

    /// Borrow the predictor datafit.
    pub fn datafit(&self) -> &D {
        &self.datafit
    }

    /// Borrow the complete penalty.
    pub fn penalty(&self) -> &P {
        &self.penalty
    }

    /// Selected training normalization policy.
    pub fn normalization(&self) -> Normalization {
        self.normalization
    }

    /// Explicit training centers, if supplied.
    pub fn centers(&self) -> Option<&[f64]> {
        self.supplied_centers.as_deref()
    }

    /// Explicit training scales, if supplied.
    pub fn scales(&self) -> Option<&[f64]> {
        self.supplied_scales.as_deref()
    }

    /// Fit with a solver that supports this composition.
    ///
    /// # Errors
    /// Forwards validation and numerical failures from the selected solver.
    pub fn fit_with<S: Solver<Self>>(&self, solver: &S) -> Result<S::Fit, S::Error> {
        solver.solve(self)
    }

    /// Select lazy training-column normalization.
    pub fn normalize(mut self, normalization: Normalization) -> Self {
        self.normalization = normalization;
        self
    }

    /// Supply centers in training-column order, checked when fitting.
    pub fn with_centers(mut self, centers: Vec<f64>) -> Self {
        self.supplied_centers = Some(centers);
        self
    }

    /// Supply positive scales in training-column order, checked when fitting.
    pub fn with_scales(mut self, scales: Vec<f64>) -> Self {
        self.supplied_scales = Some(scales);
        self
    }
}

impl<M> Solver<Problem<&M, Gaussian<'_>, L1>> for CoordinateDescent
where
    M: RawColumns<f64> + ColumnStats<f64> + ?Sized,
{
    type Fit = LassoFit;
    type Error = LassoError<M::Error>;

    fn solve(&self, problem: &Problem<&M, Gaussian<'_>, L1>) -> Result<Self::Fit, Self::Error> {
        problem.fit_coordinate(self)
    }
}

impl<M> Problem<&M, Gaussian<'_>, L1>
where
    M: RawColumns<f64> + ColumnStats<f64> + ?Sized,
{
    fn fit_coordinate(&self, solver: &CoordinateDescent) -> Result<LassoFit, LassoError<M::Error>> {
        self.penalty.validate()?;
        solver.validate(solver.criterion(self.penalty))?;
        self.datafit.validate(self.design.nrows())?;
        let prepared = self.prepare(self.datafit.fit_intercept)?;
        let solution = coordinate::solve(&prepared.matrix, self.datafit, self.penalty, solver)?;
        LassoFit::from_solution(solution, prepared.into_preprocessing())
    }
}

impl<M, D, P> Problem<&M, D, P>
where
    M: RawColumns<f64> + ColumnStats<f64> + ?Sized,
{
    pub(crate) fn prepare(
        &self,
        fit_intercept: bool,
    ) -> Result<Prepared<&M>, LassoError<M::Error>> {
        let x = self.design;
        if x.nrows() == 0 {
            return Err(invalid("training data must have at least one observation"));
        }
        validate_matrix(x)?;
        self.prepare_normalization(fit_intercept)
    }
}

impl<M, D, P> Problem<&M, D, P>
where
    M: lazymatrix::MatrixShape + ColumnStats<f64> + ?Sized,
{
    pub(crate) fn prepare_normalization(
        &self,
        fit_intercept: bool,
    ) -> Result<Prepared<&M>, LassoError<M::Error>> {
        let x = self.design;
        if x.nrows() == 0 {
            return Err(invalid("training data must have at least one observation"));
        }
        let spec = self.normalization.specification(fit_intercept);
        validate_supplied(&self.supplied_centers, x.ncols(), false)?;
        validate_supplied(&self.supplied_scales, x.ncols(), true)?;
        let center_rule = if self.supplied_centers.is_some() {
            Centering::None
        } else {
            spec.center
        };
        let scale_rule = if self.supplied_scales.is_some() {
            Scaling::None
        } else {
            spec.scale
        };
        let centered_scale = self.supplied_centers.is_some()
            && matches!(scale_rule, Scaling::L1 | Scaling::L2 | Scaling::MaxAbs);
        let stats_spec = lazymatrix::Normalization::new(
            center_rule,
            if centered_scale {
                Scaling::None
            } else {
                scale_rule
            },
        );
        let (computed_centers, mut computed_scales) =
            if stats_spec.center != Centering::None || stats_spec.scale != Scaling::None {
                x.normalization_stats(stats_spec)
                    .map_err(|source| LassoError::Backend {
                        operation: "computing training normalization statistics",
                        source,
                    })?
            } else {
                (None, None)
            };
        if centered_scale {
            let centers = self.supplied_centers.as_deref().unwrap();
            computed_scales = Some(
                match scale_rule {
                    Scaling::L1 => x.col_l1_centered(centers),
                    Scaling::L2 => x.col_l2_centered(centers),
                    Scaling::MaxAbs => x.col_maxabs_centered(centers),
                    _ => unreachable!(),
                }
                .map_err(|source| LassoError::Backend {
                    operation: "computing training normalization statistics",
                    source,
                })?,
            );
        }
        let centers = self.supplied_centers.clone().or(computed_centers);
        let mut scales = self.supplied_scales.clone().or(computed_scales);
        validate_statistics(
            &centers,
            x.ncols(),
            self.supplied_centers.is_some() || spec.center != Centering::None,
            "column centers",
        )?;
        validate_statistics(
            &scales,
            x.ncols(),
            self.supplied_scales.is_some() || spec.scale != Scaling::None,
            "column scales",
        )?;
        if let Some(scales) = &mut scales {
            for scale in scales {
                if *scale < 0.0 {
                    return Err(invalid(
                        "matrix backend returned a negative normalization scale",
                    ));
                }
                if *scale == 0.0 {
                    *scale = 1.0;
                }
            }
        }
        Ok(Prepared {
            matrix: LazyMatrix::from_parts(x, centers, scales),
            spec,
            supplied_centers: self.supplied_centers.is_some(),
            supplied_scales: self.supplied_scales.is_some(),
        })
    }
}

pub(crate) struct Prepared<M> {
    pub matrix: LazyMatrix<M>,
    spec: lazymatrix::Normalization,
    supplied_centers: bool,
    supplied_scales: bool,
}

impl<M: lazymatrix::MatrixShape> Prepared<M> {
    #[cfg(feature = "experimental-block-reader")]
    pub(crate) fn map_data<N: lazymatrix::MatrixShape>(
        self,
        transform: impl FnOnce(M) -> N,
    ) -> Prepared<N> {
        let (matrix, centers, scales) = self.matrix.into_parts();
        Prepared {
            matrix: LazyMatrix::from_parts(transform(matrix), centers, scales),
            spec: self.spec,
            supplied_centers: self.supplied_centers,
            supplied_scales: self.supplied_scales,
        }
    }

    pub fn into_preprocessing(self) -> Preprocessing {
        let (_, centers, scales) = self.matrix.into_parts();
        Preprocessing {
            spec: self.spec,
            centers,
            scales,
            supplied_centers: self.supplied_centers,
            supplied_scales: self.supplied_scales,
        }
    }
}

pub(crate) fn validate_matrix<M: RawColumns<f64> + ?Sized, E>(x: &M) -> Result<(), LassoError<E>> {
    for j in 0..x.ncols() {
        let column = x.raw_column(j);
        if column.len() != x.nrows() || column.stored_len() > x.nrows() {
            return Err(invalid(format!(
                "matrix backend returned invalid dimensions for column {j}"
            )));
        }
        let mut invalid_row = None;
        column.for_each_stored(|i, value| {
            if (i >= x.nrows() || !value.is_finite()) && invalid_row.is_none() {
                invalid_row = Some(i);
            }
        });
        if let Some(i) = invalid_row {
            return Err(invalid(format!(
                "invalid matrix entry at row {i}, column {j}: expected an in-bounds, finite value"
            )));
        }
    }
    Ok(())
}

fn validate_statistics<E>(
    values: &Option<Vec<f64>>,
    ncols: usize,
    expected: bool,
    operation: &'static str,
) -> Result<(), LassoError<E>> {
    if values.is_some() != expected || values.as_ref().is_some_and(|v| v.len() != ncols) {
        return Err(invalid(format!(
            "matrix backend returned unexpected {operation}"
        )));
    }
    if let Some(values) = values {
        for &value in values {
            finite(value, operation, 0)?;
        }
    }
    Ok(())
}

fn validate_supplied<E>(
    values: &Option<Vec<f64>>,
    ncols: usize,
    scales: bool,
) -> Result<(), LassoError<E>> {
    if let Some(values) = values {
        let name = if scales { "scales" } else { "centers" };
        if values.len() != ncols {
            return Err(invalid(format!(
                "supplied {name} length {} does not match {ncols} columns",
                values.len()
            )));
        }
        for (j, &value) in values.iter().enumerate() {
            if !value.is_finite() || (scales && value <= 0.0) {
                return Err(invalid(format!("invalid supplied {name} at column {j}")));
            }
        }
    }
    Ok(())
}
