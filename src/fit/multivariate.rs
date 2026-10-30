//! Owned multivariate parameters and joint proximal diagnostics.

use lazymatrix::{RawColumn, RawColumns};

use crate::error::{finite, invalid};
use crate::fit::back_transform_in_place;
use crate::problem::validate_matrix;
use crate::solver::proximal::Solution;
use crate::{CoefficientLayout, FitError, Preprocessing, StoppingCriterion, Termination};

/// Original-scale feature-by-response coefficients and one intercept per response.
///
/// Coefficients use [`CoefficientLayout`]'s response-major storage. The objective
/// and stopping diagnostics refer to the complete joint optimization-scale
/// problem, including all responses. No dual certificate is reported.
#[derive(Clone, Debug)]
#[must_use]
pub struct MultivariateFit {
    coefficients: Vec<f64>,
    intercepts: Vec<f64>,
    layout: CoefficientLayout,
    preprocessing: Preprocessing,
    termination: Termination,
    iterations: usize,
    objective: f64,
    stopping_criterion: StoppingCriterion,
    stopping_value: f64,
    stopping_threshold: f64,
    step: f64,
}

impl MultivariateFit {
    /// Explicit feature and response dimensions of the fitted coefficients.
    pub fn layout(&self) -> CoefficientLayout {
        self.layout
    }

    /// Original-scale coefficients in response-major order.
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    /// Original-scale feature coefficients for one response.
    pub fn coefficients_for_response(&self, response: usize) -> Option<&[f64]> {
        if response >= self.layout.nresponses() {
            return None;
        }
        let start = response * self.layout.nfeatures();
        Some(&self.coefficients[start..start + self.layout.nfeatures()])
    }

    /// Original-scale intercepts, including each response's centering correction.
    pub fn intercepts(&self) -> &[f64] {
        &self.intercepts
    }

    /// Shared training-feature normalization metadata.
    pub fn preprocessing(&self) -> &Preprocessing {
        &self.preprocessing
    }

    /// Whether the joint fit converged or exhausted its iteration budget.
    pub fn termination(&self) -> Termination {
        self.termination
    }

    /// Number of accepted joint proximal updates.
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// Final primal objective `P(beta_0, beta) = F(beta_0, beta) + R(beta)`.
    pub fn objective(&self) -> f64 {
        self.objective
    }

    /// Criterion selected for the joint fit.
    pub fn stopping_criterion(&self) -> StoppingCriterion {
        self.stopping_criterion
    }

    /// Infinity norm of the complete coefficient mapping and all free-intercept
    /// derivatives, evaluated at the returned parameters.
    pub fn stopping_value(&self) -> f64 {
        self.stopping_value
    }

    /// Absolute mapping threshold used to certify convergence.
    pub fn stopping_threshold(&self) -> f64 {
        self.stopping_threshold
    }

    /// Step that passed the final smooth quadratic-bound check.
    pub fn step_size(&self) -> f64 {
        self.step
    }

    /// Predict response columns from raw training-order features.
    ///
    /// Entry `(observation, response)` has index `response * x.nrows() + observation`.
    /// Prediction uses fitted parameters without estimating new statistics.
    /// Empty prediction batches return an empty vector.
    ///
    /// # Errors
    /// Reject feature-count mismatches, invalid entries, dimension overflow,
    /// or nonfinite arithmetic.
    pub fn predict<M: RawColumns<f64> + ?Sized>(&self, x: &M) -> Result<Vec<f64>, FitError> {
        if x.ncols() != self.layout.nfeatures() {
            return Err(invalid(
                "prediction feature count must match the fitted layout",
            ));
        }
        let len = x
            .nrows()
            .checked_mul(self.layout.nresponses())
            .ok_or_else(|| invalid("prediction dimensions overflow usize"))?;
        validate_matrix(x)?;
        let mut output = vec![0.0; len];
        for response in 0..self.layout.nresponses() {
            let column = &mut output[response * x.nrows()..(response + 1) * x.nrows()];
            column.fill(self.intercepts[response]);
            let start = response * self.layout.nfeatures();
            for (feature, &coefficient) in self.coefficients[start..start + self.layout.nfeatures()]
                .iter()
                .enumerate()
            {
                if coefficient != 0.0 {
                    x.raw_column(feature)
                        .affine_add_to(coefficient, 0.0, column);
                }
            }
            for &value in column.iter() {
                finite(value, "predicting", self.iterations)?;
            }
        }
        Ok(output)
    }

    pub(crate) fn from_solution<E>(
        mut solution: Solution,
        preprocessing: Preprocessing,
        layout: CoefficientLayout,
        fit_intercept: bool,
    ) -> Result<Self, FitError<E>> {
        let mut intercepts = if fit_intercept {
            solution.coefficients.split_off(layout.len())
        } else {
            vec![0.0; layout.nresponses()]
        };
        for (response, intercept) in intercepts.iter_mut().enumerate() {
            let start = response * layout.nfeatures();
            back_transform_in_place(
                &mut solution.coefficients[start..start + layout.nfeatures()],
                intercept,
                &preprocessing,
                solution.iterations,
            )?;
        }
        Ok(Self {
            coefficients: solution.coefficients,
            intercepts,
            layout,
            preprocessing,
            termination: solution.termination,
            iterations: solution.iterations,
            objective: solution.objective,
            stopping_criterion: solution.stopping_criterion,
            stopping_value: solution.stopping_value,
            stopping_threshold: solution.stopping_threshold,
            step: solution.step,
        })
    }
}
