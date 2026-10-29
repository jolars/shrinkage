use super::tests::matrix::Matrix;
use super::*;
use crate::{Binomial, Normalization};

fn fixture() -> Matrix {
    Matrix::from_rows(&[
        &[-1.0, 0.0001, 0.0002, -0.0001],
        &[1.0, 0.0002, -0.0001, 0.0001],
        &[-1.0, -0.0001, 0.0001, 0.0002],
        &[1.0, 0.0001, -0.0002, -0.0001],
        &[-1.0, 0.0002, -0.0001, 0.0001],
        &[1.0, -0.0001, 0.0001, 0.0002],
    ])
}

fn compare_gaussian<P: HybridPenalty + Clone>(penalty: P) {
    let x = fixture();
    let y = [-3.0, 3.0, -3.0, 3.0, -3.0, 3.0];
    let problem = Problem::new(&x, Gaussian::new(&y), penalty).normalize(Normalization::None);
    let options = Hybrid::<Vec<f64>>::new().tolerance(1e-10).check_every(1);
    let initial = crate::InitialPoint {
        coefficients: &[2.5, 0.0, 0.0, 0.0],
        intercept: 0.0,
    };
    let fit = solve_observed(&problem, &options, Some(initial), &mut NoObserver).unwrap();
    let reference = solve_observed(
        &problem,
        &options.screening(false),
        Some(initial),
        &mut NoObserver,
    )
    .unwrap();
    assert_eq!(fit.termination(), Termination::Converged);
    assert_eq!(reference.termination(), Termination::Converged);
    assert_eq!(
        fit.screening_diagnostics().screened_features,
        3,
        "{}: {:?}",
        std::any::type_name::<P>(),
        fit.screening_diagnostics()
    );
    assert_eq!(fit.screening_diagnostics().active_dimension, 1);
    assert_eq!(
        fit.screening_diagnostics().certificate_passes,
        fit.diagnostic_checks()
    );
    assert!(fit.duality_gap().unwrap() < 1e-9);
    assert!(reference.dual_certificate().is_none());
    assert!((fit.objective() - reference.objective()).abs() < 1e-12);
    assert!((fit.stopping_value() - reference.stopping_value()).abs() < 1e-10);
    for (&actual, &expected) in fit.coefficients().iter().zip(reference.coefficients()) {
        assert!((actual - expected).abs() < 1e-9);
    }
    for (j, &label) in fit.cluster_labels().iter().enumerate() {
        assert_eq!(
            fit.cluster_values()[label].abs(),
            fit.coefficients()[j].abs()
        );
    }
}

#[test]
fn gaussian_screening_retains_complete_penalties_and_full_diagnostics() {
    compare_gaussian(ClusteredLasso::new(0.3, 0.01));
    compare_gaussian(ChainFusedLasso::new(0.3, 0.02));
    compare_gaussian(Slope::new(vec![0.3, 0.25, 0.2, 0.15]));
}

fn compare_binomial<P: HybridPenalty + Clone>(penalty: P) {
    let x = fixture();
    let y = [0.1, 0.9, 0.1, 0.9, 0.1, 0.9];
    let problem = Problem::new(&x, Binomial::new(&y), penalty).normalize(Normalization::None);
    let options = Hybrid::<Vec<f64>>::new().tolerance(1e-9).check_every(1);
    let fit = problem.fit_with(&options).unwrap();
    let reference = problem.fit_with(&options.screening(false)).unwrap();
    assert_eq!(fit.termination(), Termination::Converged);
    assert_eq!(reference.termination(), Termination::Converged);
    assert_eq!(fit.screening_diagnostics().screened_features, 3);
    assert_eq!(
        fit.screening_diagnostics().certificate_passes,
        fit.outer_iterations() + 1
    );
    assert!(fit.duality_gap().unwrap() < 1e-8);
    assert!((fit.objective() - reference.objective()).abs() < 1e-12);
    for (&actual, &expected) in fit.coefficients().iter().zip(reference.coefficients()) {
        assert!((actual - expected).abs() < 1e-7);
    }
    assert!(reference.duality_gap().is_none());
}

#[test]
fn binomial_certificates_use_original_outer_iterates() {
    compare_binomial(ClusteredLasso::new(0.03, 0.001));
    compare_binomial(ChainFusedLasso::new(0.03, 0.002));
    compare_binomial(Slope::new(vec![0.03, 0.025, 0.02, 0.015]));
}

#[test]
fn screened_hybrid_products_and_aggregates_skip_matrix_entries() {
    let x = fixture();
    let y = [-3.0, 3.0, -3.0, 3.0, -3.0, 3.0];
    let penalty = ClusteredLasso::new(0.3, 0.01);
    let datafit = Gaussian::new(&y).fit_intercept(false);
    let problem = Problem::new(&x, datafit, penalty).normalize(Normalization::None);
    let mut design = NativeDesign::<_, Vec<f64>>::new_screenable(problem.prepare(false).unwrap());
    let mut work = Workspace::<ClusteredLasso>::new(y.len(), 4);
    let (objective, _) = work
        .diagnostics(&mut design, datafit, problem.penalty(), 1.0, 0)
        .unwrap();
    assert!(
        work.screening
            .check(
                &mut design,
                &datafit,
                problem.penalty(),
                &work.coefficients,
                &work.product,
                objective,
                0,
                true,
            )
            .unwrap()
    );
    assert_eq!(work.screening.active, [0]);
    x.visits.set(0);
    work.prepare(&mut design, datafit, 0).unwrap();
    let selected = x.visits.get();
    x.visits.set(0);
    work.prepare_with_selection(&mut design, datafit, 0, true)
        .unwrap();
    assert!(selected < x.visits.get());
    work.state.set_screened(&work.screening.zeroed);
    x.visits.set(0);
    work.state
        .rebuild(design.matrix(), &work.coefficients, &work.residual, 0)
        .unwrap();
    assert_eq!(x.visits.get(), y.len());
    work.state
        .sweep::<std::convert::Infallible>(
            &mut work.coefficients,
            &mut work.intercept,
            false,
            problem.penalty(),
            0,
        )
        .unwrap();
    assert_eq!(&work.coefficients[1..], [0.0; 3]);
}
