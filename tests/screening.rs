//! Original-space certificate and screening integration checks.

#[path = "common/matrix.rs"]
mod matrix;

use matrix::Matrix;
use shrinkage::{
    Binomial, ChainFusedLasso, ClusteredLasso, Gaussian, MatrixDesign, Normalization, Problem,
    ProximalFit, ProximalGradient, ProximalPenalty, RuntimeProblem, Slope, Termination,
};

struct BorrowedPenalty<'a>(&'a dyn ProximalPenalty);

impl shrinkage::Penalty for BorrowedPenalty<'_> {
    type Error = shrinkage::FitError;
    fn value(&self, coefficients: &[f64]) -> Result<f64, Self::Error> {
        self.0.value(coefficients)
    }
}

impl ProximalPenalty for BorrowedPenalty<'_> {
    fn prox(
        &self,
        input: &[f64],
        step: f64,
        output: &mut [f64],
    ) -> Result<(), shrinkage::FitError> {
        self.0.prox(input, step, output)
    }
    fn gap_safe(&self) -> Option<&dyn shrinkage::GapSafePenalty> {
        self.0.gap_safe()
    }
}

fn close(a: f64, b: f64, tolerance: f64) {
    assert!(
        (a - b).abs() <= tolerance * (1.0 + a.abs() + b.abs()),
        "{a} != {b}"
    );
}

fn fixture() -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
    let rows: Vec<Vec<f64>> = (0..24)
        .map(|i| {
            let a = (i as f64 * 0.7).sin();
            vec![
                a,
                a + (i as f64 * 1.3).cos() * 0.4,
                0.0,
                3.0,
                (i as f64 * 1.7).cos(),
            ]
        })
        .collect();
    let gaussian = rows
        .iter()
        .map(|r| 1.0 + 3.0 * r[0] + 0.02 * r[4])
        .collect();
    let binomial = rows
        .iter()
        .map(|r| 1.0 / (1.0 + (-2.0 * r[0] - 0.3).exp()))
        .collect();
    (rows, gaussian, binomial)
}

fn check_certificate(
    fit: &ProximalFit,
    rows: &[Vec<f64>],
    y: &[f64],
    penalty: &dyn ProximalPenalty,
    logistic: bool,
    intercept: bool,
) {
    let theta = fit.dual_certificate().unwrap();
    let n = y.len() as f64;
    if intercept {
        close(theta.iter().sum(), 0.0, 1e-13);
    }
    let g: Vec<f64> = (0..fit.coefficients().len())
        .map(|j| {
            let center = fit.preprocessing().centers().map_or(0.0, |v| v[j]);
            let scale = fit.preprocessing().scales().map_or(1.0, |v| v[j]);
            rows.iter()
                .zip(theta)
                .map(|(r, &theta)| (r[j] - center) / scale * theta)
                .sum()
        })
        .collect();
    assert!(
        penalty
            .gap_safe()
            .unwrap()
            .dual_feasible(&g, &vec![0.0; g.len()])
            .unwrap()
    );
    let dual = if logistic {
        y.iter()
            .zip(theta)
            .map(|(&y, &theta)| {
                assert!(theta >= (y - 1.0) / n && theta <= y / n);
                let p = (y - n * theta).clamp(0.0, 1.0);
                let entropy = |x: f64| if x == 0.0 { 0.0 } else { -x * x.ln() };
                (entropy(p) + entropy(1.0 - p)) / n
            })
            .sum()
    } else {
        y.iter()
            .zip(theta)
            .map(|(&y, &theta)| y * theta - 0.5 * n * theta * theta)
            .sum()
    };
    close(fit.dual_objective().unwrap(), dual, 1e-12);
    assert!(dual <= fit.objective() + 1e-11);
    close(fit.duality_gap().unwrap(), fit.objective() - dual, 1e-10);
}

#[test]
fn proximal_screening_matches_unscreened_gaussian_and_binomial() {
    let (rows, y, binary) = fixture();
    let matrix = Matrix::from_rows(&rows.iter().map(Vec::as_slice).collect::<Vec<_>>());
    let penalties: Vec<Box<dyn ProximalPenalty>> = vec![
        Box::new(ClusteredLasso::new(0.2, 0.01)),
        Box::new(ChainFusedLasso::new(0.1, 0.03).with_weights(vec![1.0, 0.0, 2.0, 0.5])),
        Box::new(Slope::new(vec![0.2, 0.15, 0.1, 0.05, 0.0])),
    ];
    for intercept in [false, true] {
        for normalize in [Normalization::None, Normalization::Auto] {
            for penalty in &penalties {
                let on = ProximalGradient::new().tolerance(1e-8);
                let off = on.clone().screening(false);
                let problem = Problem::new(
                    &matrix,
                    Gaussian::new(&y).fit_intercept(intercept),
                    BorrowedPenalty(&**penalty),
                )
                .normalize(normalize);
                let enabled = problem.fit_with(&on).unwrap();
                let disabled = problem.fit_with(&off).unwrap();
                assert_eq!(enabled.termination(), Termination::Converged);
                close(enabled.objective(), disabled.objective(), 1e-8);
                assert!(enabled.screening_diagnostics().screened_features > 0);
                assert!(disabled.dual_certificate().is_none());
                check_certificate(&enabled, &rows, &y, &**penalty, false, intercept);

                let problem = Problem::new(
                    &matrix,
                    Binomial::new(&binary).fit_intercept(intercept),
                    BorrowedPenalty(&**penalty),
                )
                .normalize(normalize);
                let enabled = problem.fit_with(&on).unwrap();
                let disabled = problem.fit_with(&off).unwrap();
                assert_eq!(enabled.termination(), Termination::Converged);
                close(enabled.objective(), disabled.objective(), 1e-8);
                assert!(enabled.screening_diagnostics().screened_features > 0);
                check_certificate(&enabled, &rows, &binary, &**penalty, true, intercept);
            }
        }
    }
}

#[test]
fn paths_reset_certified_zeros_and_allow_features_to_reactivate() {
    let x = Matrix::from_rows(&[&[-2.0, 0.0], &[-1.0, 0.0], &[1.0, 0.0], &[2.0, 0.0]]);
    let y = [-6.0, -3.0, 3.0, 6.0];
    let problem = Problem::new(&x, Gaussian::new(&y), Slope::new(vec![1.0, 1.0]))
        .normalize(Normalization::None);
    let solver = ProximalGradient::new().tolerance(1e-9);
    let penalties = vec![
        Slope::new(vec![10.0, 5.0]),
        Slope::new(vec![0.1, 0.05]),
        Slope::new(vec![10.0, 5.0]),
    ];
    let path = problem.fit_path_with(&solver, penalties.clone()).unwrap();
    assert_eq!(path.fits()[0].screening_diagnostics().active_dimension, 0);
    assert!(path.fits()[1].coefficients()[0] > 2.0);
    for (penalty, fit) in path.iter() {
        let cold = Problem::new(&x, Gaussian::new(&y), penalty.clone())
            .normalize(Normalization::None)
            .fit_with(&solver.clone().screening(false))
            .unwrap();
        close(fit.objective(), cold.objective(), 1e-9);
        assert!(fit.duality_gap().unwrap() < 1e-7);
    }
    assert_eq!(path.fits()[0].coefficients(), &[0.0, 0.0]);
}

#[test]
fn boxed_runtime_retains_capabilities_and_unsupported_penalties_keep_fitting() {
    let x = Matrix::from_rows(&[&[-1.0, 0.0], &[1.0, 0.0]]);
    let y = [-2.0, 2.0];
    let runtime = RuntimeProblem::new(
        Box::new(MatrixDesign::<_, Vec<f64>>::new(&x)),
        Box::new(Gaussian::new(&y)),
        Box::new(ClusteredLasso::new(0.2, 0.01)),
    );
    let fit = runtime.fit_with(&ProximalGradient::new()).unwrap();
    assert!(fit.dual_certificate().is_some());
    assert_eq!(fit.screening_diagnostics().screened_features, 1);
    let ridge = Problem::new(&x, Gaussian::new(&y), shrinkage::Ridge::new(0.1))
        .fit_with(&ProximalGradient::new())
        .unwrap();
    assert!(ridge.dual_certificate().is_none());
    assert!(!ridge.screening_diagnostics().supported);
}

#[test]
fn degenerate_strengths_use_valid_zero_certificates_without_restrictions() {
    let x = Matrix::from_rows(&[&[-1.0, -1.0], &[0.0, 0.0], &[1.0, 1.0]]);
    let y = [0.2, 0.5, 0.8];
    let penalties: Vec<Box<dyn ProximalPenalty>> = vec![
        Box::new(ClusteredLasso::new(0.0, 0.2)),
        Box::new(ChainFusedLasso::new(0.0, 0.2)),
        Box::new(Slope::new(vec![0.0; 2])),
    ];
    for penalty in penalties {
        let fit = Problem::new(&x, Binomial::new(&y), penalty)
            .fit_with(&ProximalGradient::new())
            .unwrap();
        assert_eq!(fit.dual_certificate().unwrap(), &[0.0; 3]);
        assert_eq!(fit.screening_diagnostics().screened_features, 0);
        assert!(fit.screening_diagnostics().fallback_certificates > 0);
        let expected = -y
            .iter()
            .map(|&p| (p * p.ln() + (1.0 - p) * (1.0 - p).ln()) / 3.0)
            .sum::<f64>();
        close(fit.dual_objective().unwrap(), expected, 1e-12);
    }
}

#[test]
fn binomial_candidate_projection_preserves_domain_at_extreme_logits() {
    use shrinkage::GapSafeDatafit;
    let y = [0.0, 1.0, 0.2, 1.0, 0.0];
    let datafit = Binomial::new(&y);
    let mut theta = [0.0; 5];
    datafit
        .dual_candidate(&[1000.0, -1000.0, 10.0, 5.0, -5.0], &mut theta)
        .unwrap();
    close(theta.iter().sum(), 0.0, 1e-14);
    for (&y, &theta) in y.iter().zip(&theta) {
        assert!(((y - 1.0) / 5.0..=y / 5.0).contains(&theta));
    }
    assert!(datafit.dual_objective(&theta).unwrap().is_finite());
}

#[test]
fn empty_feature_problems_keep_original_intercept_and_finite_diagnostics() {
    let x = Matrix::empty(3, 0);
    let fit = Problem::new(&x, Gaussian::new(&[1.0, 2.0, 3.0]), Slope::new(vec![]))
        .fit_with(&ProximalGradient::new())
        .unwrap();
    assert_eq!(fit.termination(), Termination::Converged);
    close(fit.intercept(), 2.0, 1e-12);
    assert_eq!(fit.screening_diagnostics().active_dimension, 0);
    assert!(fit.duality_gap().unwrap().is_finite());
}

#[test]
fn binomial_fenchel_gap_retains_dual_tails_near_boundary_probabilities() {
    use shrinkage::GapSafeDatafit;
    let loss = Binomial::new(&[1.0, 0.0]);
    let dual = [2.239470762818306e-17, -2.239470762818306e-17];
    // Independent 90-digit decimal evaluation at these represented inputs.
    let expected = 3.831466504651379e-17;
    let gap = loss.fenchel_gap(&[40.0, -37.0], &dual).unwrap();
    assert!((gap - expected).abs() < 1e-30, "{gap} != {expected}");
}

#[test]
fn gaussian_dual_objective_removes_large_free_intercept_before_summing() {
    use shrinkage::GapSafeDatafit;
    let y = [1e16, 1e16 + 2.0, 1e16 + 4.0];
    let loss = Gaussian::new(&y);
    let theta = [0.125, -0.25, 0.125];
    close(loss.dual_objective(&theta).unwrap(), -0.140625, 1e-15);
}

#[test]
fn automatic_runtime_paths_preserve_screening_at_each_strength() {
    use shrinkage::{ErasedPathPenalty, PathOptions, RuntimePathProblem};
    let x = Matrix::from_rows(&[&[-1.0, 0.0], &[1.0, 0.0]]);
    let y = [-2.0, 2.0];
    let problem = RuntimePathProblem::new(
        Box::new(MatrixDesign::<_, Vec<f64>>::new(&x)),
        Box::new(Gaussian::new(&y)),
        Box::new(ErasedPathPenalty::new(Slope::new(vec![1.0, 0.5]))),
    );
    let path = problem
        .fit_regularization_path_with(&ProximalGradient::new(), &PathOptions::new().nlambda(3))
        .unwrap();
    assert_eq!(path.len(), 3);
    for fit in path.fits() {
        assert!(fit.dual_certificate().is_some());
        assert!(fit.screening_diagnostics().supported);
        assert!(fit.duality_gap().unwrap() < 1e-8);
    }
    assert!(path.fits()[2].coefficients()[0] > 1.0);
}
