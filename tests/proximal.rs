//! Complete proximal terms and proximal-gradient composition.

#[path = "common/matrix.rs"]
mod matrix;

use matrix::Matrix;
use shrinkage::{
    CoordinateDescent, Datafit, ElasticNet, FitError, Gaussian, L1, MatrixDesign, Normalization,
    Penalty, Problem, ProximalGradient, ProximalPenalty, Ridge, RuntimeProblem, SmoothDatafit,
    StoppingCriterion, Termination,
};

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-7 * (1.0 + expected.abs()),
        "{actual} != {expected}"
    );
}

#[test]
fn zero_strength_and_elastic_net_endpoints_match() {
    let values = [-3.0, 1.0, 4.0];
    let mut output = [0.0; 3];
    for penalty in [
        ElasticNet::new(0.0, 0.0),
        ElasticNet::new(0.2, 0.0),
        ElasticNet::new(0.0, 0.7),
    ] {
        penalty.prox(&values, 0.3, &mut output).unwrap();
        let mut expected = [0.0; 3];
        L1::new(penalty.l1())
            .prox(&values, 0.3, &mut expected)
            .unwrap();
        for (value, expected) in output.iter().zip(expected) {
            close(*value, expected / (1.0 + 0.3 * penalty.l2()));
        }
        close(
            penalty.value(&values).unwrap(),
            L1::new(penalty.l1()).value(&values).unwrap()
                + Ridge::new(penalty.l2()).value(&values).unwrap(),
        );
    }
    close(Ridge::new(0.0).value(&[f64::MAX, f64::MAX]).unwrap(), 0.0);
    assert_eq!(Ridge::new(0.7).lambda(), 0.7);
}

#[test]
fn proximal_terms_and_gradients_reject_invalid_inputs() {
    let mut output = [0.0; 2];
    for invalid in [-1.0, f64::NAN, f64::INFINITY] {
        for penalty in [ElasticNet::new(invalid, 0.0), ElasticNet::new(0.0, invalid)] {
            assert!(matches!(
                penalty.prox(&[1.0, 2.0], 1.0, &mut output),
                Err(FitError::InvalidInput { .. })
            ));
            assert!(matches!(
                penalty.value(&[1.0, 2.0]),
                Err(FitError::InvalidInput { .. })
            ));
        }
    }
    for step in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(L1::new(1.0).prox(&[1.0, 2.0], step, &mut output).is_err());
    }
    assert!(Ridge::new(1.0).prox(&[1.0], 1.0, &mut output).is_err());
    assert!(
        Ridge::new(0.0)
            .prox(&[1.0, f64::NAN], 1.0, &mut output)
            .is_err()
    );
    assert!(Ridge::new(0.0).value(&[f64::INFINITY]).is_err());
    assert!(matches!(
        Ridge::new(f64::MAX).value(&[2.0]),
        Err(FitError::NumericalFailure { .. })
    ));
    let loss = Gaussian::new(&[1.0, 2.0]);
    assert!(loss.gradient(&[0.0], &mut output).is_err());
    assert!(loss.gradient(&[0.0, f64::INFINITY], &mut output).is_err());
    assert!(loss.gradient(&[0.0, 1.0], &mut [0.0]).is_err());
}

#[test]
fn proximal_lasso_agrees_with_coordinate_descent_on_correlated_columns() {
    let x = Matrix::from_rows(&[
        &[0.0, 1.0, 5.0],
        &[2.0, 0.0, 5.0],
        &[4.0, 3.0, 5.0],
        &[6.0, 1.0, 5.0],
    ]);
    let y = [1.0, 2.0, -1.0, 4.0];
    for intercept in [false, true] {
        let problem = Problem::new(
            &x,
            Gaussian::new(&y).fit_intercept(intercept),
            L1::new(0.15),
        );
        let proximal = problem
            .fit_with(&ProximalGradient::<Vec<f64>>::new().tolerance(1e-9))
            .unwrap();
        let coordinate = problem
            .fit_with(&CoordinateDescent::new().tolerance(1e-10))
            .unwrap();
        assert_eq!(proximal.termination(), Termination::Converged);
        close(proximal.objective(), coordinate.objective());
        for (a, b) in proximal
            .predict(&x)
            .unwrap()
            .iter()
            .zip(coordinate.predict(&x).unwrap())
        {
            close(*a, b);
        }
    }
}

#[test]
fn lazy_normalization_preserves_the_complete_penalty_and_predictions() {
    let x = Matrix::from_rows(&[&[0.0, 1.0], &[2.0, 0.0], &[4.0, 3.0], &[6.0, 1.0]]);
    let normalized = Matrix::from_rows(&[&[-1.5, -0.5], &[-0.5, -2.5], &[0.5, 3.5], &[1.5, -0.5]]);
    let y = [1.0, 2.0, -1.0, 4.0];
    let penalty = ElasticNet::new(0.15, 0.3);
    let solver = ProximalGradient::<Vec<f64>>::new().tolerance(1e-9);
    for intercept in [false, true] {
        let datafit = Gaussian::new(&y).fit_intercept(intercept);
        let lazy = Problem::new(&x, datafit, penalty)
            .with_centers(vec![3.0, 1.25])
            .with_scales(vec![2.0, 0.5])
            .fit_with(&solver)
            .unwrap();
        let explicit = Problem::new(&normalized, datafit, penalty)
            .normalize(Normalization::None)
            .fit_with(&solver)
            .unwrap();
        assert_eq!(lazy.termination(), Termination::Converged);
        close(lazy.objective(), explicit.objective());
        let theta: Vec<_> = lazy
            .coefficients()
            .iter()
            .zip([2.0, 0.5])
            .map(|(b, s)| b * s)
            .collect();
        close(
            lazy.objective(),
            datafit.value(&lazy.predict(&x).unwrap()).unwrap() + penalty.value(&theta).unwrap(),
        );
        for (a, b) in lazy
            .predict(&x)
            .unwrap()
            .iter()
            .zip(explicit.predict(&normalized).unwrap())
        {
            close(*a, b);
        }
        assert!(lazy.preprocessing().centers_were_supplied());
        assert!(lazy.preprocessing().scales_were_supplied());
        if !intercept {
            close(
                lazy.intercept(),
                -3.0 * lazy.coefficients()[0] - 1.25 * lazy.coefficients()[1],
            );
        }
    }
}

#[test]
fn iteration_limit_reports_mapping_at_returned_parameters() {
    let x = Matrix::from_rows(&[&[1.0, 0.0], &[2.0, 1.0], &[0.0, 4.0]]);
    let y = [1.0, 4.0, 2.0];
    let penalty = ElasticNet::new(0.2, 0.3);
    let fit = Problem::new(&x, Gaussian::new(&y), penalty)
        .normalize(Normalization::None)
        .fit_with(
            &ProximalGradient::<Vec<f64>>::new()
                .initial_step(0.05)
                .max_iterations(1)
                .tolerance(1e-12),
        )
        .unwrap();
    assert_eq!(fit.termination(), Termination::IterationLimit);
    assert_eq!(fit.iterations(), 1);
    let predictions = fit.predict(&x).unwrap();
    let derivatives: Vec<_> = predictions
        .iter()
        .zip(y)
        .map(|(eta, y)| (eta - y) / 3.0)
        .collect();
    let gradient = [
        derivatives[0] + 2.0 * derivatives[1],
        derivatives[1] + 4.0 * derivatives[2],
    ];
    let input: Vec<_> = fit
        .coefficients()
        .iter()
        .zip(gradient)
        .map(|(b, g)| b - fit.step_size() * g)
        .collect();
    let mut candidate = [0.0; 2];
    penalty
        .prox(&input, fit.step_size(), &mut candidate)
        .unwrap();
    let mapping = fit
        .coefficients()
        .iter()
        .zip(candidate)
        .map(|(b, next)| ((b - next) / fit.step_size()).abs())
        .fold(derivatives.iter().sum::<f64>().abs(), f64::max);
    close(fit.stopping_value(), mapping);
    close(
        fit.objective(),
        Gaussian::new(&y).value(&predictions).unwrap() + penalty.value(fit.coefficients()).unwrap(),
    );
    assert_eq!(
        fit.stopping_criterion(),
        StoppingCriterion::proximal_gradient_mapping(1e-12)
    );
}

#[test]
fn empty_features_zero_columns_and_unpenalized_fits_work() {
    let empty = Matrix::empty(3, 0);
    for intercept in [false, true] {
        let fit = Problem::new(
            &empty,
            Gaussian::new(&[2.0, 3.0, 4.0]).fit_intercept(intercept),
            Ridge::new(1.0),
        )
        .fit_with(&ProximalGradient::<Vec<f64>>::new())
        .unwrap();
        assert_eq!(fit.termination(), Termination::Converged);
        close(fit.intercept(), if intercept { 3.0 } else { 0.0 });
    }
    let x = Matrix::from_rows(&[&[0.0, 5.0, 0.0], &[1.0, 5.0, 0.0], &[2.0, 5.0, 0.0]]);
    let fit = Problem::new(&x, Gaussian::new(&[1.0, 3.0, 5.0]), Ridge::new(0.0))
        .fit_with(&ProximalGradient::<Vec<f64>>::new().tolerance(1e-9))
        .unwrap();
    close(fit.coefficients()[0], 2.0);
    close(fit.coefficients()[1], 0.0);
    close(fit.coefficients()[2], 0.0);
    close(fit.intercept(), 1.0);
    assert!(fit.predict(&Matrix::empty(0, 3)).unwrap().is_empty());
    assert!(fit.predict(&Matrix::empty(3, 2)).is_err());
}

#[test]
fn solver_validates_options_and_reports_line_search_failure() {
    let x = Matrix::from_rows(&[&[0.0], &[1.0], &[2.0]]);
    let problem = Problem::new(&x, Gaussian::new(&[1.0, 3.0, 5.0]), Ridge::new(0.1));
    for solver in [
        ProximalGradient::<Vec<f64>>::new().tolerance(0.0),
        ProximalGradient::new().tolerance(f64::NAN),
        ProximalGradient::new().initial_step(-1.0),
        ProximalGradient::new().initial_step(f64::INFINITY),
        ProximalGradient::new().max_iterations(0),
    ] {
        assert!(matches!(
            problem.fit_with(&solver),
            Err(FitError::InvalidInput { .. })
        ));
    }
    assert!(matches!(
        problem.fit_with(
            &ProximalGradient::<Vec<f64>>::new()
                .initial_step(1e6)
                .max_backtracks(0)
        ),
        Err(FitError::LineSearchFailure { .. })
    ));
    assert!(
        Problem::new(&x, Gaussian::new(&[1.0]), Ridge::new(0.1))
            .fit_with(&ProximalGradient::<Vec<f64>>::new())
            .is_err()
    );
    assert!(
        Problem::new(&x, Gaussian::new(&[1.0, f64::NAN, 2.0]), Ridge::new(0.1))
            .fit_with(&ProximalGradient::<Vec<f64>>::new())
            .is_err()
    );
    assert!(
        Problem::new(&x, Gaussian::new(&[1.0, 2.0, 3.0]), Ridge::new(-1.0))
            .fit_with(&ProximalGradient::<Vec<f64>>::new())
            .is_err()
    );
    assert!(
        Problem::new(&x, Gaussian::new(&[1.0, 2.0, 3.0]), Ridge::new(0.1))
            .with_scales(vec![0.0])
            .fit_with(&ProximalGradient::<Vec<f64>>::new())
            .is_err()
    );
    let empty = Matrix::empty(0, 1);
    assert!(
        Problem::new(&empty, Gaussian::new(&[]), Ridge::new(0.0))
            .fit_with(&ProximalGradient::<Vec<f64>>::new())
            .is_err()
    );
    let bad = Matrix::from_rows(&[&[f64::NAN]]);
    assert!(
        Problem::new(&bad, Gaussian::new(&[1.0]), Ridge::new(0.0))
            .fit_with(&ProximalGradient::<Vec<f64>>::new())
            .is_err()
    );
    assert!(
        Problem::new(&x, Gaussian::new(&[1.0, 2.0, 3.0]), L1::new(0.1))
            .fit_with(
                &CoordinateDescent::new()
                    .terminate_on(StoppingCriterion::proximal_gradient_mapping(1e-6))
            )
            .is_err()
    );
}

// A coupled quadratic has a complete prox that preserves the input mean.
struct CenteredQuadratic;

impl Penalty for CenteredQuadratic {
    type Error = FitError;
    fn value(&self, coefficients: &[f64]) -> Result<f64, FitError> {
        let mean = coefficients.iter().sum::<f64>() / coefficients.len() as f64;
        Ok(0.5 * coefficients.iter().map(|b| (b - mean).powi(2)).sum::<f64>())
    }
}

impl ProximalPenalty for CenteredQuadratic {
    fn prox(&self, input: &[f64], step: f64, output: &mut [f64]) -> Result<(), FitError> {
        let mean = input.iter().sum::<f64>() / input.len() as f64;
        for (out, value) in output.iter_mut().zip(input) {
            *out = (value + step * mean) / (1.0 + step);
        }
        Ok(())
    }
}

struct TwiceGaussian<'a>(Gaussian<'a>);

impl Datafit for TwiceGaussian<'_> {
    type Error = FitError;
    fn value(&self, predictor: &[f64]) -> Result<f64, FitError> {
        Ok(2.0 * self.0.value(predictor)?)
    }
}

impl SmoothDatafit for TwiceGaussian<'_> {
    fn nobs(&self) -> usize {
        self.0.nobs()
    }
    fn fits_intercept(&self) -> bool {
        self.0.fits_intercept()
    }
    fn gradient(&self, predictor: &[f64], output: &mut [f64]) -> Result<(), FitError> {
        self.0.gradient(predictor, output)?;
        for value in output {
            *value *= 2.0;
        }
        Ok(())
    }
}

#[test]
fn downstream_smooth_loss_and_nonseparable_complete_prox_compose() {
    let x = Matrix::from_rows(&[&[-1.0, -1.0], &[-1.0, 1.0], &[1.0, -1.0], &[1.0, 1.0]]);
    let y = [-4.0, -2.0, 2.0, 4.0];
    let fit = Problem::new(
        &x,
        TwiceGaussian(Gaussian::new(&y).fit_intercept(false)),
        CenteredQuadratic,
    )
    .normalize(Normalization::None)
    .fit_with(&ProximalGradient::<Vec<f64>>::new().tolerance(1e-9))
    .unwrap();
    assert_eq!(fit.termination(), Termination::Converged);
    close(fit.coefficients()[0], 8.0 / 3.0);
    close(fit.coefficients()[1], 4.0 / 3.0);
}

struct FailingProducts {
    matrix: Matrix,
    forward_calls: std::cell::Cell<usize>,
    fail_forward: usize,
    fail_transpose: bool,
    fail_statistics: bool,
}

impl shrinkage::lazymatrix::MatrixShape for FailingProducts {
    fn nrows(&self) -> usize {
        shrinkage::lazymatrix::MatrixShape::nrows(&self.matrix)
    }
    fn ncols(&self) -> usize {
        shrinkage::lazymatrix::MatrixShape::ncols(&self.matrix)
    }
}

impl shrinkage::lazymatrix::MatrixErrorType for FailingProducts {
    type Error = std::io::Error;
}

impl shrinkage::lazymatrix::RawColumns<f64> for FailingProducts {
    type Column<'a> = matrix::Column<'a>;
    fn raw_column(&self, j: usize) -> Self::Column<'_> {
        shrinkage::lazymatrix::RawColumns::raw_column(&self.matrix, j)
    }
}

macro_rules! delegate_stats {
    ($($name:ident $(($arg:ident))?),* $(,)?) => {
        $(fn $name(&self $(, $arg: &[f64])?) -> Result<Vec<f64>, std::io::Error> {
            if self.fail_statistics {
                return Err(std::io::Error::other("injected statistics failure"));
            }
            Ok(shrinkage::lazymatrix::ColumnStats::$name(&self.matrix $(, $arg)?).unwrap())
        })*
    };
}

impl shrinkage::lazymatrix::ColumnStats<f64> for FailingProducts {
    delegate_stats!(
        col_means,
        col_sds,
        col_mins,
        col_ranges,
        col_maxabs,
        col_l1,
        col_l2,
        col_l2_centered(centers),
        col_l1_centered(centers),
        col_maxabs_centered(centers)
    );
}

impl shrinkage::lazymatrix::MatVecInto<Vec<f64>> for FailingProducts {
    fn matvec_into(&self, input: &Vec<f64>, output: &mut Vec<f64>) -> Result<(), std::io::Error> {
        let call = self.forward_calls.get() + 1;
        self.forward_calls.set(call);
        if call == self.fail_forward {
            output.fill(f64::NAN);
            return Err(std::io::Error::other("injected forward failure"));
        }
        shrinkage::lazymatrix::MatVecInto::matvec_into(&self.matrix, input, output).unwrap();
        Ok(())
    }
}

impl shrinkage::lazymatrix::MatTransposeVecInto<Vec<f64>> for FailingProducts {
    fn mat_transpose_vec_into(
        &self,
        input: &Vec<f64>,
        output: &mut Vec<f64>,
    ) -> Result<(), std::io::Error> {
        if self.fail_transpose {
            output.fill(f64::NAN);
            return Err(std::io::Error::other("injected transpose failure"));
        }
        shrinkage::lazymatrix::MatTransposeVecInto::mat_transpose_vec_into(
            &self.matrix,
            input,
            output,
        )
        .unwrap();
        Ok(())
    }
}

#[test]
fn product_errors_preserve_sources_and_stop_after_partial_writes() {
    use std::error::Error;
    for (fail_forward, fail_transpose, operation) in [
        (1, false, "computing a forward design product"),
        (3, false, "computing a forward design product"),
        (usize::MAX, true, "computing a transposed design product"),
    ] {
        let matrix = FailingProducts {
            matrix: Matrix::from_rows(&[&[0.0], &[1.0], &[2.0]]),
            forward_calls: std::cell::Cell::new(0),
            fail_forward,
            fail_transpose,
            fail_statistics: false,
        };
        let error = Problem::new(&matrix, Gaussian::new(&[1.0, 3.0, 5.0]), Ridge::new(0.1))
            .fit_with(&ProximalGradient::<Vec<f64>>::new().initial_step(0.1))
            .unwrap_err();
        assert!(error.source().unwrap().to_string().contains("injected"));
        assert!(
            matches!(error, FitError::Backend { operation: actual, .. } if actual == operation)
        );
        if !fail_transpose {
            assert_eq!(matrix.forward_calls.get(), fail_forward);
        }
    }
}

#[test]
fn runtime_backend_errors_retain_original_sources_and_discard_partial_outputs() {
    use std::error::Error;
    for (fail_forward, fail_transpose, fail_statistics, operation) in [
        (1, false, false, "computing a forward design product"),
        (3, false, false, "computing a forward design product"),
        (
            usize::MAX,
            true,
            false,
            "computing a transposed design product",
        ),
        (
            usize::MAX,
            false,
            true,
            "computing training normalization statistics",
        ),
    ] {
        let matrix = FailingProducts {
            matrix: Matrix::from_rows(&[&[0.0], &[1.0], &[2.0]]),
            forward_calls: std::cell::Cell::new(0),
            fail_forward,
            fail_transpose,
            fail_statistics,
        };
        let error = RuntimeProblem::new(
            Box::new(MatrixDesign::<_, Vec<f64>>::new(&matrix)),
            Box::new(Gaussian::new(&[1.0, 3.0, 5.0])),
            Box::new(Ridge::new(0.1)),
        )
        .fit_with(&ProximalGradient::new().initial_step(0.1))
        .unwrap_err();
        let original = error.source().unwrap().source().unwrap();
        assert!(original.downcast_ref::<std::io::Error>().is_some());
        assert!(original.to_string().contains("injected"));
        assert!(
            matches!(error, FitError::Backend { operation: actual, .. } if actual == operation)
        );
        if fail_statistics {
            assert_eq!(matrix.forward_calls.get(), 0);
        } else if !fail_transpose {
            assert_eq!(matrix.forward_calls.get(), fail_forward);
        }
    }
}

#[test]
fn nonfinite_arithmetic_is_distinct_from_invalid_input() {
    let matrix = Matrix::from_rows(&[&[0.0]]);
    assert!(matches!(
        Problem::new(&matrix, Gaussian::new(&[f64::MAX]), Ridge::new(0.0))
            .normalize(Normalization::None)
            .fit_with(&ProximalGradient::<Vec<f64>>::new()),
        Err(FitError::NumericalFailure { .. })
    ));
    let matrix = Matrix::from_rows(&[&[f64::MAX]]);
    assert!(matches!(
        Problem::new(
            &matrix,
            Gaussian::new(&[2.0]).fit_intercept(false),
            Ridge::new(0.0)
        )
        .normalize(Normalization::None)
        .fit_with(&ProximalGradient::<Vec<f64>>::new()),
        Err(FitError::NumericalFailure { .. })
    ));
}

#[test]
fn complete_elastic_net_prox_satisfies_its_optimality_equations() {
    let penalty = ElasticNet::new(0.3, 0.7);
    let input = [-3.0, -0.1, 0.0, 0.1, 4.0];
    let mut output = [0.0; 5];
    penalty.prox(&input, 0.5, &mut output).unwrap();
    for (&v, &z) in input.iter().zip(&output) {
        let smooth_derivative = (z - v) / 0.5 + 0.7 * z;
        if z == 0.0 {
            assert!(smooth_derivative.abs() <= 0.3);
        } else {
            close(smooth_derivative + 0.3 * z.signum(), 0.0);
        }
    }
    close(
        penalty.value(&[-2.0, 3.0]).unwrap(),
        0.3 * 5.0 + 0.35 * 13.0,
    );
    Ridge::new(0.7).prox(&input, 0.5, &mut output).unwrap();
    close(output[0], -3.0 / 1.35);
    L1::new(0.3).prox(&input, 0.5, &mut output).unwrap();
    close(output[0], -2.85);
}

#[test]
fn gaussian_predictor_gradient_matches_finite_differences() {
    let loss = Gaussian::new(&[1.0, 4.0, -2.0]);
    let predictor = [2.0, 2.0, 0.0];
    let mut gradient = [0.0; 3];
    loss.gradient(&predictor, &mut gradient).unwrap();
    for j in 0..3 {
        let mut plus = predictor;
        let mut minus = predictor;
        plus[j] += 1e-5;
        minus[j] -= 1e-5;
        close(
            gradient[j],
            (loss.value(&plus).unwrap() - loss.value(&minus).unwrap()) / 2e-5,
        );
    }
}

#[test]
fn orthogonal_ridge_and_elastic_net_have_analytic_solutions() {
    let x = Matrix::from_rows(&[&[-1.0, -1.0], &[-1.0, 1.0], &[1.0, -1.0], &[1.0, 1.0]]);
    let y = [2.0, -4.0, 6.0, 0.0];
    let solver = ProximalGradient::<Vec<f64>>::new()
        .tolerance(1e-9)
        .initial_step(100.0);
    let ridge = Problem::new(&x, Gaussian::new(&y), Ridge::new(0.5))
        .normalize(Normalization::None)
        .fit_with(&solver)
        .unwrap();
    let net = Problem::new(&x, Gaussian::new(&y), ElasticNet::new(0.25, 0.5))
        .normalize(Normalization::None)
        .fit_with(&solver)
        .unwrap();
    for fit in [&ridge, &net] {
        assert_eq!(fit.termination(), Termination::Converged);
        close(fit.intercept(), 1.0);
        assert!(fit.stopping_value() <= fit.stopping_threshold());
    }
    close(ridge.coefficients()[0], 2.0 / 1.5);
    close(ridge.coefficients()[1], -3.0 / 1.5);
    close(net.coefficients()[0], 1.75 / 1.5);
    close(net.coefficients()[1], -2.75 / 1.5);
}
