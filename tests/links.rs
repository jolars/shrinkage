//! Link composition, derivative capabilities, and Fisher-scoring fits.

#[path = "common/matrix.rs"]
mod matrix;

use matrix::Matrix;
use shrinkage::{
    Binomial, ConvexLink, Datafit, DiagonalHessianDatafit, DifferentiableDatafit, ElasticNet,
    FitError, Gaussian, Identity, Irls, L1, Link, Logit, Normalization, NullModelDatafit,
    PathOptions, Probit, Problem, QuadraticModelDatafit, Ridge, Termination,
    TwiceDifferentiableLink,
};

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance * (1.0 + expected.abs()),
        "{actual} != {expected}"
    );
}

#[test]
fn built_in_links_round_trip_and_supply_two_derivatives() {
    fn check<L: TwiceDifferentiableLink>(link: L, means: &[f64]) {
        for &mu in means {
            let eta = link.link(mu).unwrap();
            close(link.inverse_link(eta).unwrap(), mu, 2e-14);
            let step = 1e-5;
            let derivative = (link.inverse_link(eta + step).unwrap()
                - link.inverse_link(eta - step).unwrap())
                / (2.0 * step);
            close(link.inverse_link_derivative(eta).unwrap(), derivative, 1e-9);
            let second = (link.inverse_link_derivative(eta + step).unwrap()
                - link.inverse_link_derivative(eta - step).unwrap())
                / (2.0 * step);
            close(
                link.inverse_link_second_derivative(eta).unwrap(),
                second,
                1e-9,
            );
        }
    }
    check(Identity, &[-3.0, 0.0, 4.0]);
    check(Logit, &[0.001, 0.2, 0.5, 0.8, 0.999]);
    check(Probit, &[0.001, 0.2, 0.5, 0.8, 0.999]);
    for invalid in [-0.1, 0.0, 1.0, 1.1, f64::NAN] {
        assert!(Logit.link(invalid).is_err());
        assert!(Probit.link(invalid).is_err());
    }
}

#[derive(Clone, Copy)]
struct ExponentialLink;

impl Link for ExponentialLink {
    fn link(&self, mu: f64) -> Result<f64, FitError> {
        if !mu.is_finite() || mu <= 0.0 {
            return Err(FitError::InvalidInput {
                message: "mean must be positive".into(),
            });
        }
        Ok(mu.ln())
    }
    fn inverse_link(&self, eta: f64) -> Result<f64, FitError> {
        Ok(eta.exp())
    }
    fn inverse_link_derivative(&self, eta: f64) -> Result<f64, FitError> {
        Ok(eta.exp())
    }
}

impl TwiceDifferentiableLink for ExponentialLink {
    fn inverse_link_second_derivative(&self, eta: f64) -> Result<f64, FitError> {
        Ok(eta.exp())
    }
}

#[test]
fn observed_hessian_can_be_negative_while_fisher_curvature_is_positive() {
    let datafit = Gaussian::new(&[3.0]).with_link(ExponentialLink);
    let mut observed = [0.0];
    let mut fisher = [0.0];
    datafit.hessian_diagonal(&[0.0], &mut observed).unwrap();
    datafit.working_curvature(&[0.0], &mut fisher).unwrap();
    assert_eq!(observed, [-1.0]);
    assert_eq!(fisher, [1.0]);
    let x = Matrix::empty(1, 0);
    let fit = Problem::new(&x, datafit, Ridge::new(0.0))
        .fit_from_with_observer(
            &Irls::new().tolerance(1e-11),
            shrinkage::InitialPoint {
                coefficients: &[],
                intercept: 0.0,
            },
            &mut |_: shrinkage::SolverStep<'_>| {},
        )
        .unwrap();
    assert_eq!(fit.termination(), Termination::Converged);
    close(fit.intercept(), 3.0_f64.ln(), 1e-10);
}

#[test]
fn probit_intercept_and_automatic_path_use_the_selected_link() {
    let x = Matrix::empty(4, 0);
    let datafit = Binomial::new(&[0.0, 1.0, 1.0, 1.0]).with_link(Probit);
    let expected = Probit.link(0.75).unwrap();
    close(datafit.null_intercepts().unwrap()[0], expected, 1e-14);
    let problem = Problem::new(&x, datafit, Ridge::new(0.1));
    let fit = problem.fit_with(&Irls::new().tolerance(1e-11)).unwrap();
    assert_eq!(fit.termination(), Termination::Converged);
    close(fit.intercept(), expected, 1e-9);
    close(
        problem
            .datafit()
            .inverse_link(fit.predict(&x).unwrap()[0])
            .unwrap(),
        0.75,
        1e-9,
    );
    let x = Matrix::from_rows(&[&[-2.0], &[-1.0], &[1.0], &[2.0]]);
    let problem = Problem::new(&x, datafit, L1::new(1.0));
    let path = problem
        .fit_regularization_path_with(
            &Irls::new().tolerance(1e-10),
            &PathOptions::new().nlambda(4).early_stopping(false),
        )
        .unwrap();
    assert_eq!(path.len(), 4);
    assert!(
        path.fits()
            .iter()
            .all(|fit| fit.termination() == Termination::Converged)
    );
}

#[test]
fn probit_matches_independent_high_precision_fixtures() {
    let x = Matrix::from_rows(&[&[-2.0], &[-1.0], &[0.0], &[0.5], &[1.0], &[2.0]]);
    let y = [0.05, 0.2, 0.3, 0.65, 0.8, 0.95];
    for line in include_str!("fixtures/probit.csv").lines().skip(1) {
        let fields: Vec<_> = line.split(',').collect();
        let values: Vec<f64> = fields[1..].iter().map(|s| s.parse().unwrap()).collect();
        if fields[0] == "tail" {
            let y = [values[1]];
            let datafit = Binomial::new(&y).with_link(Probit);
            let eta = [values[0]];
            let mut gradient = [0.0];
            let mut hessian = [0.0];
            let mut fisher = [0.0];
            datafit.gradient(&eta, &mut gradient).unwrap();
            datafit.hessian_diagonal(&eta, &mut hessian).unwrap();
            datafit.working_curvature(&eta, &mut fisher).unwrap();
            for (actual, expected) in [
                datafit.value(&eta).unwrap(),
                gradient[0],
                hessian[0],
                fisher[0],
            ]
            .into_iter()
            .zip(&values[2..])
            {
                close(actual, *expected, 2e-10);
                if *expected != 0.0 && expected.abs() < 1e-10 {
                    close(actual / expected, 1.0, 2e-10);
                }
            }
        } else {
            let problem = Problem::new(
                &x,
                Binomial::new(&y).with_link(Probit),
                ElasticNet::new(values[0], values[1]),
            )
            .normalize(Normalization::None);
            let fit = problem
                .fit_with(&Irls::new().tolerance(1e-11).max_iterations(500))
                .unwrap();
            assert_eq!(fit.termination(), Termination::Converged);
            close(fit.objective(), values[2], 1e-9);
            close(fit.intercept(), values[3], 1e-9);
            close(fit.coefficients()[0], values[4], 1e-9);
        }
    }
}

struct ScaledLogit(f64);
impl Link for ScaledLogit {
    fn validate(&self) -> Result<(), FitError> {
        if !self.0.is_finite() || self.0 <= 0.0 {
            return Err(FitError::InvalidInput {
                message: "scale must be positive".into(),
            });
        }
        Ok(())
    }
    fn link(&self, mu: f64) -> Result<f64, FitError> {
        Ok(self.0 * Logit.link(mu)?)
    }
    fn inverse_link(&self, eta: f64) -> Result<f64, FitError> {
        Logit.inverse_link(eta / self.0)
    }
    fn inverse_link_derivative(&self, eta: f64) -> Result<f64, FitError> {
        Ok(Logit.inverse_link_derivative(eta / self.0)? / self.0)
    }
}
impl ConvexLink<Binomial<'_>> for ScaledLogit {}

#[test]
fn a_custom_link_needs_no_second_derivative_or_copy_for_paths() {
    let x = Matrix::from_rows(&[&[-2.0], &[-1.0], &[1.0], &[2.0]]);
    let y = [0.1, 0.3, 0.7, 0.9];
    let datafit = Binomial::new(&y)
        .fit_intercept(false)
        .with_link(ScaledLogit(2.0));
    assert!(!datafit.fits_intercept());
    let problem = Problem::new(&x, datafit, L1::new(1.0)).normalize(Normalization::None);
    let solver = Irls::new().tolerance(1e-10);
    let penalties = vec![L1::new(0.1), L1::new(0.01), L1::new(0.1)];
    let path = problem.fit_path_with(&solver, penalties).unwrap();
    for (fit, penalty) in path.fits().iter().zip(path.penalties()) {
        let independent = Problem::new(
            &x,
            Binomial::new(&y)
                .with_link(ScaledLogit(2.0))
                .fit_intercept(false),
            *penalty,
        )
        .normalize(Normalization::None)
        .fit_with(&solver)
        .unwrap();
        assert_eq!(fit.termination(), Termination::Converged);
        close(fit.objective(), independent.objective(), 1e-9);
        close(fit.coefficients()[0], independent.coefficients()[0], 1e-8);
    }
    let automatic = problem
        .fit_regularization_path_with(
            &solver,
            &PathOptions::new().nlambda(3).early_stopping(false),
        )
        .unwrap();
    assert_eq!(automatic.len(), 3);
    assert!(automatic.fits()[0].coefficients().iter().all(|&b| b == 0.0));
    assert!(
        Problem::new(
            &x,
            Binomial::new(&y).with_link(ScaledLogit(0.0)),
            L1::new(0.1)
        )
        .fit_with(&solver)
        .is_err()
    );
}

#[test]
fn probit_null_intercept_preserves_a_small_failure_total() {
    let y = [1.0, f64::from_bits(1.0_f64.to_bits() - 1)];
    let eta = Binomial::new(&y)
        .with_link(Probit)
        .null_intercepts()
        .unwrap()[0];
    let (_, log_q) = Probit.log_probabilities(eta).unwrap();
    close(log_q, ((1.0 - y[1]) / 2.0).ln(), 1e-14);
    assert_eq!(Probit.link(0.5).unwrap(), 0.0);
}

struct RestrictedLink {
    rejected: std::cell::Cell<usize>,
}
impl Link for RestrictedLink {
    fn link(&self, mu: f64) -> Result<f64, FitError> {
        Ok(mu.exp() + 1.0)
    }
    fn inverse_link(&self, eta: f64) -> Result<f64, FitError> {
        Ok((eta - 1.0).ln())
    }
    fn inverse_link_derivative(&self, eta: f64) -> Result<f64, FitError> {
        Ok(1.0 / (eta - 1.0))
    }
    fn valid_eta(&self, eta: f64) -> bool {
        let valid = eta.is_finite() && eta > 1.0;
        if !valid {
            self.rejected.set(self.rejected.get() + 1);
        }
        valid
    }
}

#[test]
fn restricted_domains_backtrack_and_supplied_initial_points_take_precedence() {
    let x = Matrix::empty(1, 0);
    let problem = Problem::new(
        &x,
        Gaussian::new(&[-4.0]).with_link(RestrictedLink { rejected: 0.into() }),
        Ridge::new(0.0),
    );
    let mut initial = None;
    let mut observe = |step: shrinkage::SolverStep<'_>| {
        if step.kind == shrinkage::StepKind::Initial {
            initial = Some(step.intercept);
        }
    };
    let fit = problem
        .fit_from_with_observer(
            &Irls::new().tolerance(1e-8),
            shrinkage::InitialPoint {
                coefficients: &[],
                intercept: 2.0,
            },
            &mut observe,
        )
        .unwrap();
    assert_eq!(initial, Some(2.0));
    assert_eq!(fit.termination(), Termination::Converged);
    close(fit.intercept(), 1.0 + (-4.0_f64).exp(), 1e-9);
    assert!(problem.datafit().link_function().rejected.get() > 0);

    let x = Matrix::from_rows(&[&[2.0]]);
    let problem = Problem::new(
        &x,
        Gaussian::new(&[0.0])
            .with_link(RestrictedLink { rejected: 0.into() })
            .fit_intercept(false),
        Ridge::new(0.0),
    )
    .normalize(Normalization::None);
    assert!(matches!(
        problem.fit_with(&Irls::new()),
        Err(FitError::InvalidInput { .. })
    ));
    let fit = problem
        .fit_from_with_observer(
            &Irls::new(),
            shrinkage::InitialPoint {
                coefficients: &[1.0],
                intercept: 0.0,
            },
            &mut |_: shrinkage::SolverStep<'_>| {},
        )
        .unwrap();
    assert_eq!(fit.termination(), Termination::Converged);
}

#[test]
fn linked_defaults_agree_with_specialized_losses_and_derivatives() {
    let y = [0.0, 0.3, 1.0];
    let eta = [-40.0, 0.0, 40.0];
    let mut actual = [0.0; 3];
    let mut expected = [0.0; 3];
    let original = Binomial::new(&y);
    let linked = original.with_link(Logit);
    close(
        linked.value(&eta).unwrap(),
        original.value(&eta).unwrap(),
        1e-14,
    );
    original.gradient(&eta, &mut expected).unwrap();
    linked.gradient(&eta, &mut actual).unwrap();
    for (a, b) in actual.into_iter().zip(expected) {
        close(a / b, 1.0, 1e-13);
    }
    original.hessian_diagonal(&eta, &mut expected).unwrap();
    linked.working_curvature(&eta, &mut actual).unwrap();
    for (a, b) in actual.into_iter().zip(expected) {
        close(a / b, 1.0, 1e-13);
    }
    let original = Gaussian::new(&y);
    let linked = original.with_link(Identity);
    close(
        linked.value(&eta).unwrap(),
        original.value(&eta).unwrap(),
        1e-14,
    );
    original.gradient(&eta, &mut expected).unwrap();
    linked.gradient(&eta, &mut actual).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn linked_derivatives_match_differences_and_validate_inputs() {
    for y in [0.0, 0.3, 1.0] {
        let response = [y];
        let datafit = Binomial::new(&response).with_link(Probit);
        for eta in [-3.0, -0.5, 0.0, 0.5, 3.0] {
            let step = 1e-5;
            let derivative = (datafit.value(&[eta + step]).unwrap()
                - datafit.value(&[eta - step]).unwrap())
                / (2.0 * step);
            let mut gradient = [0.0];
            datafit.gradient(&[eta], &mut gradient).unwrap();
            close(gradient[0], derivative, 1e-9);
            let mut plus = [0.0];
            let mut minus = [0.0];
            let mut hessian = [0.0];
            datafit.gradient(&[eta + step], &mut plus).unwrap();
            datafit.gradient(&[eta - step], &mut minus).unwrap();
            datafit.hessian_diagonal(&[eta], &mut hessian).unwrap();
            close(hessian[0], (plus[0] - minus[0]) / (2.0 * step), 1e-9);
        }
        assert!(datafit.value(&[]).is_err());
        assert!(datafit.gradient(&[0.0], &mut []).is_err());
        assert!(datafit.working_curvature(&[f64::NAN], &mut [0.0]).is_err());
        assert!(datafit.hessian_diagonal(&[0.0], &mut []).is_err());
    }
    assert!(
        Binomial::new(&[2.0])
            .with_link(Probit)
            .value(&[0.0])
            .is_err()
    );
    assert!(
        Gaussian::new(&[f64::NAN])
            .with_link(Identity)
            .value(&[0.0])
            .is_err()
    );
    let x = Matrix::empty(4, 0);
    let datafit: Box<dyn QuadraticModelDatafit> =
        Box::new(Binomial::new(&[0.0, 1.0, 1.0, 1.0]).with_link(Probit));
    let fit = Problem::new(&x, datafit, Ridge::new(0.0))
        .fit_with(&Irls::new())
        .unwrap();
    close(fit.intercept(), Probit.link(0.75).unwrap(), 1e-6);
}

struct CubicLogit;
impl Link for CubicLogit {
    fn link(&self, mu: f64) -> Result<f64, FitError> {
        Ok(Logit.link(mu)?.cbrt())
    }
    fn inverse_link(&self, eta: f64) -> Result<f64, FitError> {
        Logit.inverse_link(eta.powi(3))
    }
    fn inverse_link_derivative(&self, eta: f64) -> Result<f64, FitError> {
        Ok(3.0 * eta * eta * Logit.inverse_link_derivative(eta.powi(3))?)
    }
}
impl TwiceDifferentiableLink for CubicLogit {
    fn inverse_link_second_derivative(&self, eta: f64) -> Result<f64, FitError> {
        Ok(6.0 * eta * Logit.inverse_link_derivative(eta.powi(3))?
            + 9.0 * eta.powi(4) * Logit.inverse_link_second_derivative(eta.powi(3))?)
    }
}

#[test]
fn an_exact_hessian_handles_a_zero_inverse_derivative() {
    let datafit = Binomial::new(&[0.3]).with_link(CubicLogit);
    let mut hessian = [1.0];
    datafit.hessian_diagonal(&[0.0], &mut hessian).unwrap();
    assert_eq!(hessian, [0.0]);
}

#[test]
fn gaussian_identity_automatic_paths_match_the_default_family() {
    let x = Matrix::from_rows(&[&[-1.0], &[0.0], &[1.0], &[2.0]]);
    let y = [1.0, 2.0, 3.0, 4.0];
    let solver = Irls::new().tolerance(1e-10);
    let original = Problem::new(&x, Gaussian::new(&y), L1::new(1.0)).normalize(Normalization::None);
    let linked = Problem::new(&x, Gaussian::new(&y).with_link(Identity), L1::new(1.0))
        .normalize(Normalization::None);
    assert_eq!(
        linked.maximum_penalty_with(&solver).unwrap(),
        original.maximum_penalty_with(&solver).unwrap()
    );
    let options = PathOptions::new().nlambda(4).early_stopping(false);
    let expected = original
        .fit_regularization_path_with(&solver, &options)
        .unwrap();
    let actual = linked
        .fit_regularization_path_with(&solver, &options)
        .unwrap();
    assert_eq!(actual.lambdas(), expected.lambdas());
    for (actual, expected) in actual.fits().iter().zip(expected.fits()) {
        assert_eq!(actual.termination(), Termination::Converged);
        close(actual.objective(), expected.objective(), 1e-12);
        close(actual.intercept(), expected.intercept(), 1e-12);
        close(actual.coefficients()[0], expected.coefficients()[0], 1e-12);
    }
}
