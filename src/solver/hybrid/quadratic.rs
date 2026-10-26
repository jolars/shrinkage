//! Quadratic models retain gradients without dividing by row curvatures.

use lazymatrix::LogicalColumn;

use crate::error::{finite, invalid};
use crate::{Datafit, FitError, Gaussian, SmoothDatafit};

#[derive(Clone, Copy)]
pub(super) enum Model<'a> {
    Gaussian(Gaussian<'a>),
    Newton {
        anchor: &'a [f64],
        gradient: &'a [f64],
        weights: &'a [f64],
        fit_intercept: bool,
    },
}

impl<'a> From<Gaussian<'a>> for Model<'a> {
    fn from(value: Gaussian<'a>) -> Self {
        Self::Gaussian(value)
    }
}

impl Model<'_> {
    pub(super) fn weights(&self) -> &[f64] {
        match self {
            Self::Gaussian(_) => &[],
            Self::Newton { weights, .. } => weights,
        }
    }

    pub(super) fn validate(&self, n: usize) -> Result<(), FitError> {
        match self {
            Self::Gaussian(datafit) => datafit.validate(n),
            Self::Newton {
                anchor,
                gradient,
                weights,
                ..
            } => {
                if n == 0 || anchor.len() != n || gradient.len() != n || weights.len() != n {
                    return Err(invalid(
                        "quadratic model dimensions must match nonempty rows",
                    ));
                }
                if anchor.iter().chain(*gradient).any(|x| !x.is_finite())
                    || weights.iter().any(|x| !x.is_finite() || *x < 0.0)
                {
                    return Err(invalid(
                        "quadratic anchors and gradients must be finite and weights finite and nonnegative",
                    ));
                }
                Ok(())
            }
        }
    }

    pub(super) fn refresh<E>(
        self,
        prediction: &mut [f64],
        intercept: f64,
        derivative: &mut [f64],
        iteration: usize,
    ) -> Result<f64, FitError<E>> {
        let n = prediction.len() as f64;
        if let Self::Gaussian(datafit) = self {
            for ((r, d), &y) in prediction.iter_mut().zip(derivative).zip(datafit.response) {
                *r = finite(y - *r - intercept, "refreshing hybrid residuals", iteration)?;
                *d = -*r / n;
            }
            return Ok(finite(
                prediction.iter().map(|r| r * r / (2.0 * n)).sum(),
                "computing quadratic loss",
                iteration,
            )?);
        }
        for eta in prediction.iter_mut() {
            *eta = finite(
                *eta + intercept,
                "refreshing quadratic predictors",
                iteration,
            )?;
        }
        let loss = self
            .value(prediction)
            .map_err(|e| FitError::from_component(e, iteration))?;
        self.gradient(prediction, derivative)
            .map_err(|e| FitError::from_component(e, iteration))?;
        for (r, &g) in prediction.iter_mut().zip(derivative.iter()) {
            *r = finite(-n * g, "refreshing quadratic gradient residuals", iteration)?;
        }
        Ok(loss)
    }
}

impl Datafit for Model<'_> {
    type Error = FitError;

    fn value(&self, predictor: &[f64]) -> Result<f64, FitError> {
        match self {
            Self::Gaussian(datafit) => datafit.value(predictor),
            Self::Newton {
                anchor,
                gradient,
                weights,
                ..
            } => {
                self.validate(predictor.len())?;
                let n = predictor.len() as f64;
                Ok(finite(
                    predictor
                        .iter()
                        .zip(*anchor)
                        .zip(*gradient)
                        .zip(*weights)
                        .map(|(((&eta, &base), &g), &w)| {
                            let d = eta - base;
                            g * d + 0.5 * (w * d) * d / n
                        })
                        .sum(),
                    "computing quadratic loss",
                    0,
                )?)
            }
        }
    }
}

impl SmoothDatafit for Model<'_> {
    fn nobs(&self) -> usize {
        match self {
            Self::Gaussian(datafit) => datafit.nobs(),
            Self::Newton { anchor, .. } => anchor.len(),
        }
    }

    fn fits_intercept(&self) -> bool {
        match self {
            Self::Gaussian(datafit) => datafit.fits_intercept(),
            Self::Newton { fit_intercept, .. } => *fit_intercept,
        }
    }

    fn gradient(&self, predictor: &[f64], output: &mut [f64]) -> Result<(), FitError> {
        if output.len() != predictor.len() {
            return Err(invalid("quadratic gradient dimensions must match"));
        }
        match self {
            Self::Gaussian(datafit) => datafit.gradient(predictor, output),
            Self::Newton {
                anchor,
                gradient,
                weights,
                ..
            } => {
                self.validate(predictor.len())?;
                let n = predictor.len() as f64;
                for ((((out, &eta), &base), &g), &w) in output
                    .iter_mut()
                    .zip(predictor)
                    .zip(*anchor)
                    .zip(*gradient)
                    .zip(*weights)
                {
                    *out = finite(
                        g + w * ((eta - base) / n),
                        "computing a quadratic gradient",
                        0,
                    )?;
                }
                Ok(())
            }
        }
    }
}

/// Weights are `n * H[i]`, so empty storage preserves Gaussian arithmetic.
#[derive(Default)]
pub(super) struct Weights {
    values: Vec<f64>,
}

impl Weights {
    pub(super) fn set(&mut self, values: &[f64]) {
        self.values.clear();
        self.values.extend_from_slice(values);
    }

    pub(super) fn column_curvature(&self, column: impl LogicalColumn<f64>) -> f64 {
        let n = column.len() as f64;
        if self.values.is_empty() {
            column.norm_squared() / n
        } else {
            column.weighted_norm_squared(&self.values) / n
        }
    }

    pub(super) fn intercept_curvature(&self, n: usize) -> f64 {
        if self.values.is_empty() {
            1.0
        } else {
            self.values.iter().map(|w| w / n as f64).sum()
        }
    }

    pub(super) fn curvature<E>(
        &self,
        column: &[f64],
        iteration: usize,
    ) -> Result<f64, FitError<E>> {
        for &a in column {
            finite(a, "forming a cluster aggregate", iteration)?;
        }
        let n = column.len() as f64;
        let (q, nonzero) = if self.values.is_empty() {
            (
                column.iter().map(|a| a * a / n).sum(),
                column.iter().any(|&a| a != 0.0),
            )
        } else {
            (
                column
                    .iter()
                    .zip(&self.values)
                    .map(|(&a, &w)| (w * a) * a / n)
                    .sum(),
                column
                    .iter()
                    .zip(&self.values)
                    .any(|(&a, &w)| a != 0.0 && w != 0.0),
            )
        };
        let q = finite(q, "computing weighted cluster curvature", iteration)?;
        if q == 0.0 && nonzero {
            return Err(FitError::NumericalFailure {
                operation: "computing cluster curvature without underflow",
                iteration,
            });
        }
        Ok(q)
    }

    pub(super) fn linear<E>(
        &self,
        column: &[f64],
        residual: &[f64],
        value: f64,
        iteration: usize,
    ) -> Result<f64, FitError<E>> {
        let n = residual.len() as f64;
        let b = if self.values.is_empty() {
            column
                .iter()
                .zip(residual)
                .map(|(&a, &r)| a * (r + a * value) / n)
                .sum()
        } else {
            column
                .iter()
                .zip(residual)
                .zip(&self.values)
                .map(|((&a, &r), &w)| a * (r + w * a * value) / n)
                .sum()
        };
        Ok(finite(
            b,
            "forming a weighted cluster linear term",
            iteration,
        )?)
    }

    pub(super) fn subtract<E>(
        &self,
        column: &[f64],
        residual: &mut [f64],
        delta: f64,
        iteration: usize,
    ) -> Result<(), FitError<E>> {
        if self.values.is_empty() {
            for (&a, r) in column.iter().zip(residual) {
                *r = finite(*r - a * delta, "updating cluster residuals", iteration)?;
            }
        } else {
            for ((&a, r), &w) in column.iter().zip(residual).zip(&self.values) {
                *r = finite(
                    *r - (w * a) * delta,
                    "updating weighted cluster residuals",
                    iteration,
                )?;
            }
        }
        Ok(())
    }

    pub(super) fn update_intercept<E>(
        &self,
        residual: &mut [f64],
        intercept: &mut f64,
        iteration: usize,
    ) -> Result<(), FitError<E>> {
        let delta = if self.values.is_empty() {
            residual.iter().map(|r| r / residual.len() as f64).sum()
        } else {
            let sum = finite(
                residual.iter().sum(),
                "summing weighted residuals",
                iteration,
            )?;
            let weight = finite(
                self.values.iter().sum(),
                "summing intercept weights",
                iteration,
            )?;
            if weight == 0.0 && sum == 0.0 {
                return Ok(());
            }
            sum / weight
        };
        let delta = finite(delta, "minimizing the weighted hybrid intercept", iteration)?;
        *intercept = finite(
            *intercept + delta,
            "updating the hybrid intercept",
            iteration,
        )?;
        if self.values.is_empty() {
            for r in residual {
                *r = finite(*r - delta, "updating intercept residuals", iteration)?;
            }
        } else {
            for (r, &w) in residual.iter_mut().zip(&self.values) {
                *r = finite(
                    *r - w * delta,
                    "updating weighted intercept residuals",
                    iteration,
                )?;
            }
        }
        Ok(())
    }

    pub(super) fn image_norm<E>(
        &self,
        product: &[f64],
        intercept: f64,
        iteration: usize,
    ) -> Result<f64, FitError<E>> {
        let mut norm: f64 = 0.0;
        if self.values.is_empty() {
            for &a in product {
                norm = norm.hypot(finite(
                    a + intercept,
                    "checking the hybrid quadratic model",
                    iteration,
                )?);
            }
        } else {
            for (&a, &w) in product.iter().zip(&self.values) {
                norm = norm.hypot(finite(
                    w.sqrt() * (a + intercept),
                    "checking the weighted hybrid quadratic model",
                    iteration,
                )?);
            }
        }
        Ok(norm / (product.len() as f64).sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::hybrid::tests::matrix::Matrix;
    use crate::solver::hybrid::{HybridPenalty, Workspace, iterate_quadratic};
    use crate::solver::proximal::NativeDesign;
    use crate::{
        ChainFusedLasso, ClusteredLasso, Hybrid, Normalization, Penalty, Problem, ProximalGradient,
        ProximalPenalty, Slope, Termination,
    };

    struct WithIntercept<P>(P);

    impl<P: ProximalPenalty> Penalty for WithIntercept<P> {
        type Error = FitError;

        fn value(&self, coefficients: &[f64]) -> Result<f64, FitError> {
            self.0.value(&coefficients[..coefficients.len() - 1])
        }
    }

    impl<P: ProximalPenalty> ProximalPenalty for WithIntercept<P> {
        fn prox(&self, input: &[f64], step: f64, output: &mut [f64]) -> Result<(), FitError> {
            let p = input.len() - 1;
            self.0.prox(&input[..p], step, &mut output[..p])?;
            output[p] = input[p];
            Ok(())
        }
    }

    fn compare<P: HybridPenalty + Clone>(penalty: P) {
        let rows = [
            [1.0, 2.0, -1.0],
            [0.0, 1.0, 2.0],
            [2.0, 0.0, 1.0],
            [1.0, -1.0, 1.0],
            [-1.0, 2.0, -2.0],
        ];
        let y = [-1.0, 0.5, 2.0, 1.0, -2.0];
        let x = Matrix::from_rows(&rows.iter().map(|r| r.as_slice()).collect::<Vec<_>>());
        let mut work = Workspace::<P>::new(5, 3);
        for weights in [[0.0_f64, 0.2, 2.0, 0.5, 1.3], [0.0; 5], [1.0; 5]] {
            for intercept in [false, true] {
                let weighted: Vec<Vec<_>> = rows
                    .iter()
                    .zip(weights)
                    .map(|(row, w)| {
                        row.iter()
                            .map(|a| a * w.sqrt())
                            .chain([if intercept { w.sqrt() } else { 0.0 }])
                            .collect()
                    })
                    .collect();
                let explicit =
                    Matrix::from_rows(&weighted.iter().map(Vec::as_slice).collect::<Vec<_>>());
                let response: Vec<_> = y.iter().zip(weights).map(|(y, w)| y * w.sqrt()).collect();
                let expected = Problem::new(
                    &explicit,
                    Gaussian::new(&response).fit_intercept(false),
                    WithIntercept(penalty.clone()),
                )
                .normalize(Normalization::None)
                .fit_with(
                    &ProximalGradient::<Vec<f64>>::new()
                        .tolerance(1e-10)
                        .max_iterations(100_000),
                )
                .unwrap();
                assert_eq!(expected.termination(), Termination::Converged);
                let problem = Problem::new(
                    &x,
                    Gaussian::new(&y).fit_intercept(intercept),
                    penalty.clone(),
                )
                .normalize(Normalization::None);
                let mut design =
                    NativeDesign::<_, Vec<f64>>::new(problem.prepare(intercept).unwrap());
                let initial = [0.8, -0.6, 0.2];
                work.coefficients.copy_from_slice(&initial);
                work.intercept = if intercept { 0.4 } else { 0.0 };
                let anchor: Vec<_> = rows
                    .iter()
                    .map(|row| {
                        work.intercept + row.iter().zip(initial).map(|(a, b)| a * b).sum::<f64>()
                    })
                    .collect();
                let gradient: Vec<_> = anchor
                    .iter()
                    .zip(y)
                    .zip(weights)
                    .map(|((&eta, y), w)| w * (eta - y) / 5.0)
                    .collect();
                let constant: f64 = anchor
                    .iter()
                    .zip(y)
                    .zip(weights)
                    .map(|((&eta, y), w)| w * (eta - y).powi(2) / 10.0)
                    .sum();
                let model = Model::Newton {
                    anchor: &anchor,
                    gradient: &gradient,
                    weights: &weights,
                    fit_intercept: intercept,
                };
                let proximal = Problem::new(&x, model, penalty.clone())
                    .normalize(Normalization::None)
                    .fit_with(
                        &ProximalGradient::<Vec<f64>>::new()
                            .tolerance(1e-10)
                            .max_iterations(100_000),
                    )
                    .unwrap();
                assert_eq!(proximal.termination(), Termination::Converged);
                assert!((proximal.objective() + constant - expected.objective()).abs() < 1e-9);
                for policy in [crate::HybridStep::default(), crate::HybridStep::FixedBound] {
                    work.coefficients.copy_from_slice(&initial);
                    work.intercept = if intercept { 0.4 } else { 0.0 };
                    let report = iterate_quadratic(
                        &mut design,
                        model,
                        &penalty,
                        &Hybrid::<()>::new()
                            .step_policy(policy)
                            .max_iterations(100_000),
                        1e-10,
                        &mut work,
                    )
                    .unwrap();
                    assert_eq!(report.termination, Termination::Converged);
                    assert!((report.objective + constant - expected.objective()).abs() < 1e-9);
                    if weights.iter().any(|&w| w > 0.0) {
                        for (&a, &b) in work
                            .coefficients
                            .iter()
                            .chain([&work.intercept])
                            .zip(expected.coefficients())
                        {
                            assert!((a - b).abs() < 1e-7, "{a} != {b}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn weighted_subproblems_match_explicit_row_scaling() {
        compare(ClusteredLasso::new(0.07, 0.03));
        compare(ChainFusedLasso::new(0.07, 0.03).with_weights(vec![0.0, 2.0]));
        compare(Slope::new(vec![0.1, 0.07, 0.0]));
    }

    #[test]
    fn weighted_summaries_and_zero_curvatures_are_exact() {
        let mut weights = Weights::default();
        weights.set(&[0.0, 2.0, 1.0]);
        let column = [100.0, 2.0, -1.0];
        let mut residual = [0.0, 4.0, 2.0];
        assert_eq!(
            weights
                .curvature::<std::convert::Infallible>(&column, 0)
                .unwrap(),
            3.0
        );
        assert_eq!(
            weights
                .linear::<std::convert::Infallible>(&column, &residual, 1.0, 0)
                .unwrap(),
            5.0
        );
        weights
            .subtract::<std::convert::Infallible>(&column, &mut residual, 0.5, 0)
            .unwrap();
        assert_eq!(residual, [0.0, 2.0, 2.5]);
        let mut intercept = 0.0;
        weights
            .update_intercept::<std::convert::Infallible>(&mut residual, &mut intercept, 0)
            .unwrap();
        assert_eq!(intercept, 1.5);
        assert_eq!(residual, [0.0, -1.0, 1.0]);
        weights.set(&[0.0; 3]);
        assert_eq!(
            weights
                .curvature::<std::convert::Infallible>(&column, 0)
                .unwrap(),
            0.0
        );
        weights
            .update_intercept::<std::convert::Infallible>(&mut [0.0; 3], &mut intercept, 0)
            .unwrap();
        assert_eq!(intercept, 1.5);
        weights.set(&[1e-300]);
        assert!(
            weights
                .curvature::<std::convert::Infallible>(&[1e-100], 0)
                .is_err()
        );
    }
}
