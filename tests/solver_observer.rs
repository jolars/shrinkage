//! Accepted solver observations, numerical equivalence, and coefficient scales.

#![cfg(feature = "nalgebra_v0_34")]

use nalgebra::{DMatrix, DVector};
use shrinkage::{
    Binomial, ChainFusedLasso, ClusteredLasso, CoordinateDescent, Datafit, ElasticNet, Gaussian,
    Hybrid, Irls, L1, Normalization, Penalty, Problem, ProximalGradient, Ridge, Slope, SolverStep,
    StepKind,
};

macro_rules! check_trace {
    ($matrix:expr, $datafit:expr, $penalty:expr, $solver:expr) => {{
        let problem = Problem::new(&$matrix, $datafit, $penalty).normalize(Normalization::None);
        let solver = $solver;
        let expected = problem.fit_with(&solver).unwrap();
        let mut steps = Vec::new();
        let observed = problem
            .fit_with_observer(&solver, &mut |step: SolverStep<'_>| {
                let eta: Vec<_> = (0..$matrix.nrows())
                    .map(|i| {
                        step.intercept
                            + (0..$matrix.ncols())
                                .map(|j| $matrix[(i, j)] * step.coefficients[j])
                                .sum::<f64>()
                    })
                    .collect();
                let objective = problem.datafit().value(&eta).unwrap()
                    + problem.penalty().value(step.coefficients).unwrap();
                steps.push((
                    step.kind,
                    step.iteration,
                    step.coefficients.to_vec(),
                    step.intercept,
                    objective,
                ));
            })
            .unwrap();
        assert_eq!(observed.coefficients(), expected.coefficients());
        assert_eq!(observed.intercept(), expected.intercept());
        assert_eq!(observed.objective(), expected.objective());
        assert_eq!(observed.iterations(), expected.iterations());
        assert_eq!(observed.stopping_value(), expected.stopping_value());
        assert_eq!(observed.termination(), expected.termination());
        assert_eq!(steps.first().unwrap().0, StepKind::Initial);
        assert_eq!(
            steps.last().unwrap().0,
            StepKind::Final(observed.termination())
        );
        assert_eq!(steps.last().unwrap().1, observed.iterations());
        assert_eq!(steps.last().unwrap().2, observed.coefficients());
        for pair in steps.windows(2) {
            assert!(pair[0].1 <= pair[1].1);
            assert!(
                pair[1].4 <= pair[0].4 + 1e-12,
                "accepted iterates must descend: {pair:?}"
            );
        }
        steps
    }};
}

#[test]
fn observed_fits_preserve_all_solver_results_and_only_emit_accepted_iterates() {
    let x = DMatrix::from_row_slice(4, 2, &[1., 0.5, -1., -0.5, 0., 1., 0., -1.]);
    let y = [1.5, -1.5, -1., 1.];
    let binary = [0.8, 0.2, 0.3, 0.7];
    let steps = check_trace!(
        x,
        Gaussian::new(&y).fit_intercept(false),
        L1::new(0.05),
        CoordinateDescent::new()
    );
    for pair in steps.windows(2) {
        if matches!(pair[1].0, StepKind::Coordinate { .. }) {
            assert!(
                pair[0]
                    .2
                    .iter()
                    .zip(&pair[1].2)
                    .filter(|(a, b)| a != b)
                    .count()
                    <= 1
            );
        }
    }
    check_trace!(
        x,
        Gaussian::new(&y).fit_intercept(false),
        Ridge::new(0.1),
        ProximalGradient::<DVector<f64>>::new()
    );
    check_trace!(
        x,
        Binomial::new(&binary).fit_intercept(false),
        ElasticNet::new(0.1, 0.5),
        Irls::new()
    );
    check_trace!(
        x,
        Gaussian::new(&y).fit_intercept(false),
        ClusteredLasso::new(0.05, 0.05),
        Hybrid::<DVector<f64>>::new()
    );
    check_trace!(
        x,
        Gaussian::new(&y).fit_intercept(false),
        ChainFusedLasso::new(0.05, 0.05),
        Hybrid::<DVector<f64>>::new()
    );
    check_trace!(
        x,
        Binomial::new(&binary).fit_intercept(false),
        Slope::new(vec![0.05, 0.025]),
        Hybrid::<DVector<f64>>::new()
    );
}

#[test]
fn initially_optimal_and_iteration_limited_fits_have_final_events() {
    let x = DMatrix::from_row_slice(2, 2, &[1., 0.5, 0., 1.]);
    let y = [1., -1.];
    let steps = check_trace!(
        x,
        Gaussian::new(&y).fit_intercept(false),
        L1::new(10.),
        CoordinateDescent::new()
    );
    assert_eq!(steps.len(), 2);
    check_trace!(
        x,
        Gaussian::new(&y).fit_intercept(false),
        Ridge::new(0.01),
        ProximalGradient::<DVector<f64>>::new().max_iterations(1)
    );
    check_trace!(
        x,
        Gaussian::new(&y).fit_intercept(false),
        L1::new(0.01),
        CoordinateDescent::new().max_iterations(1)
    );
    check_trace!(
        x,
        Gaussian::new(&y).fit_intercept(false),
        Ridge::new(0.01),
        Irls::new().max_iterations(1).max_inner_iterations(1)
    );
    check_trace!(
        x,
        Gaussian::new(&y).fit_intercept(false),
        Slope::new(vec![0.01, 0.005]),
        Hybrid::<DVector<f64>>::new().max_iterations(1)
    );
}

#[test]
fn observations_use_optimization_scale_and_include_intercept_updates() {
    let x = DMatrix::from_row_slice(4, 2, &[2., 5., 4., 7., 6., 4., 8., 8.]);
    let y = [3., 7., 11., 15.];
    let problem = Problem::new(&x, Gaussian::new(&y), L1::new(0.1));
    let mut final_state = None;
    let mut intercept_updates = 0;
    let fit = problem
        .fit_with_observer(&CoordinateDescent::new(), &mut |step: SolverStep<'_>| {
            if step.kind == StepKind::Intercept {
                intercept_updates += 1;
            }
            if matches!(step.kind, StepKind::Final(_)) {
                final_state = Some((step.coefficients.to_vec(), step.intercept));
            }
        })
        .unwrap();
    assert!(intercept_updates > 0);
    let (coefficients, intercept) = final_state.unwrap();
    let centers = fit.preprocessing().centers().unwrap();
    let scales = fit.preprocessing().scales().unwrap();
    for j in 0..2 {
        assert!((coefficients[j] / scales[j] - fit.coefficients()[j]).abs() < 1e-14);
    }
    let original_intercept = intercept
        - (0..2)
            .map(|j| centers[j] * coefficients[j] / scales[j])
            .sum::<f64>();
    assert!((original_intercept - fit.intercept()).abs() < 1e-14);
}

macro_rules! check_initial {
    ($matrix:expr, $datafit:expr, $penalty:expr, $solver:expr, $normalization:expr) => {{
        let problem = Problem::new(&$matrix, $datafit, $penalty).normalize($normalization);
        let solver = $solver;
        let expected = problem.fit_with(&solver).unwrap();
        let initial = shrinkage::InitialPoint {
            coefficients: &[-1.3, 2.1],
            intercept: if problem.datafit().fits_intercept() {
                0.7
            } else {
                0.0
            },
        };
        let mut first = None;
        let fit = problem
            .fit_from_with_observer(&solver, initial, &mut |step: SolverStep<'_>| {
                if first.is_none() {
                    first = Some((step.kind, step.coefficients.to_vec(), step.intercept));
                }
            })
            .unwrap();
        let first = first.unwrap();
        assert_eq!(first.0, StepKind::Initial);
        assert_eq!(first.1, initial.coefficients);
        assert_eq!(first.2, initial.intercept);
        assert_eq!(fit.termination(), shrinkage::Termination::Converged);
        assert!((fit.objective() - expected.objective()).abs() < 1e-7);
        for (&actual, &expected) in fit.coefficients().iter().zip(expected.coefficients()) {
            assert!((actual - expected).abs() < 1e-4);
        }
        for initial in [
            shrinkage::InitialPoint {
                coefficients: &[1.],
                intercept: 0.,
            },
            shrinkage::InitialPoint {
                coefficients: &[f64::NAN, 0.],
                intercept: 0.,
            },
            shrinkage::InitialPoint {
                coefficients: &[0., 0.],
                intercept: f64::INFINITY,
            },
        ] {
            let mut calls = 0;
            assert!(
                problem
                    .fit_from_with_observer(&solver, initial, &mut |_: SolverStep<'_>| calls += 1)
                    .is_err()
            );
            assert_eq!(calls, 0);
        }
    }};
}

#[test]
fn explicit_initial_points_reach_the_same_optimum_in_every_kernel() {
    let x = DMatrix::from_row_slice(4, 2, &[1., 0.5, -1., -0.5, 0., 1., 0., -1.]);
    let y = [1.5, -1.5, -1., 1.];
    let binary = [0.8, 0.2, 0.3, 0.7];
    for intercept in [false, true] {
        for normalization in [Normalization::None, Normalization::Standardize] {
            check_initial!(
                x,
                Gaussian::new(&y).fit_intercept(intercept),
                L1::new(0.05),
                CoordinateDescent::new(),
                normalization
            );
            check_initial!(
                x,
                Gaussian::new(&y).fit_intercept(intercept),
                Ridge::new(0.1),
                ProximalGradient::<DVector<f64>>::new(),
                normalization
            );
            check_initial!(
                x,
                Binomial::new(&binary).fit_intercept(intercept),
                ElasticNet::new(0.05, 0.1),
                Irls::new(),
                normalization
            );
            check_initial!(
                x,
                Gaussian::new(&y).fit_intercept(intercept),
                ClusteredLasso::new(0.05, 0.05),
                Hybrid::<DVector<f64>>::new(),
                normalization
            );
            check_initial!(
                x,
                Binomial::new(&binary).fit_intercept(intercept),
                Slope::new(vec![0.05, 0.025]),
                Hybrid::<DVector<f64>>::new(),
                normalization
            );
        }
    }
}

#[test]
fn explicit_nonzero_intercepts_are_rejected_when_disabled() {
    let x = DMatrix::identity(2, 2);
    let y = [1., -1.];
    let problem = Problem::new(&x, Gaussian::new(&y).fit_intercept(false), L1::new(0.1))
        .normalize(Normalization::None);
    let error = problem
        .fit_from_with_observer(
            &CoordinateDescent::new(),
            shrinkage::InitialPoint {
                coefficients: &[0., 0.],
                intercept: 1.,
            },
            &mut |_: SolverStep<'_>| {},
        )
        .unwrap_err();
    assert!(error.to_string().contains("initial point"));
}

struct RecordedStep {
    kind: StepKind,
    outer: Option<usize>,
    iteration: usize,
    coefficients: Vec<f64>,
    intercept: f64,
}

#[derive(Default)]
struct InnerTrace {
    events: Vec<RecordedStep>,
}

impl shrinkage::SolverObserver for InnerTrace {
    fn observe(&mut self, step: SolverStep<'_>) {
        self.events.push(RecordedStep {
            kind: step.kind,
            outer: None,
            iteration: step.iteration,
            coefficients: step.coefficients.to_vec(),
            intercept: step.intercept,
        });
    }

    fn observe_inner(&mut self, step: SolverStep<'_>, outer_iteration: usize) {
        self.events.push(RecordedStep {
            kind: step.kind,
            outer: Some(outer_iteration),
            iteration: step.iteration,
            coefficients: step.coefficients.to_vec(),
            intercept: step.intercept,
        });
    }
}

macro_rules! check_inner_trace {
    ($matrix:expr, $datafit:expr, $penalty:expr, $solver:expr, $kind:pat) => {{
        let problem = Problem::new(&$matrix, $datafit, $penalty).normalize(Normalization::None);
        let solver = $solver;
        let expected = problem.fit_with(&solver).unwrap();
        let mut trace = InnerTrace::default();
        let actual = problem.fit_with_observer(&solver, &mut trace).unwrap();
        assert_eq!(actual.coefficients(), expected.coefficients());
        assert_eq!(actual.intercept(), expected.intercept());
        assert_eq!(actual.objective(), expected.objective());
        assert_eq!(actual.iterations(), expected.iterations());
        assert_eq!(actual.stopping_value(), expected.stopping_value());
        assert_eq!(actual.termination(), expected.termination());
        assert!(
            trace
                .events
                .iter()
                .any(|event| event.outer.is_some() && matches!(event.kind, $kind))
        );
        for RecordedStep {
            kind,
            outer,
            iteration,
            coefficients,
            intercept,
        } in &trace.events
        {
            if let Some(outer) = outer {
                assert!(*outer > 0 && *iteration > 0);
                assert!(!matches!(kind, StepKind::Initial | StepKind::Final(_)));
            }
            let eta: Vec<_> = (0..$matrix.nrows())
                .map(|i| {
                    intercept
                        + (0..$matrix.ncols())
                            .map(|j| $matrix[(i, j)] * coefficients[j])
                            .sum::<f64>()
                })
                .collect();
            let objective = problem.datafit().value(&eta).unwrap()
                + problem.penalty().value(coefficients).unwrap();
            assert!(objective.is_finite());
        }
        for pair in trace.events.windows(2) {
            if pair[1].outer.is_some() && matches!(pair[1].kind, StepKind::Coordinate { .. }) {
                assert_eq!(
                    pair[0]
                        .coefficients
                        .iter()
                        .zip(&pair[1].coefficients)
                        .filter(|(a, b)| a != b)
                        .count(),
                    1
                );
                assert_eq!(pair[0].intercept, pair[1].intercept);
            }
        }
        assert_eq!(trace.events.last().unwrap().outer, None);
        assert_eq!(
            trace.events.last().unwrap().coefficients,
            actual.coefficients()
        );
        assert_eq!(trace.events.last().unwrap().intercept, actual.intercept());
    }};
}

#[test]
fn inner_observations_expose_real_updates_without_changing_fits() {
    let x = DMatrix::from_row_slice(4, 2, &[1., 0.5, -1., -0.5, 0., 1., 0., -1.]);
    let y = [1.5, -1.5, -1., 1.];
    let binary = [0.8, 0.2, 0.3, 0.7];
    check_inner_trace!(
        x,
        Gaussian::new(&y),
        L1::new(0.01),
        Irls::new(),
        StepKind::Coordinate { .. }
    );
    check_inner_trace!(
        x,
        Binomial::new(&binary),
        L1::new(0.01),
        Irls::new(),
        StepKind::Coordinate { .. }
    );
    check_inner_trace!(
        x,
        Gaussian::new(&y),
        ClusteredLasso::new(0.005, 0.005),
        Hybrid::<DVector<f64>>::new(),
        StepKind::Block { .. }
    );
    check_inner_trace!(
        x,
        Gaussian::new(&y),
        ChainFusedLasso::new(0.005, 0.005),
        Hybrid::<DVector<f64>>::new(),
        StepKind::Block { .. }
    );
    check_inner_trace!(
        x,
        Gaussian::new(&y),
        Slope::new(vec![0.01, 0.005]),
        Hybrid::<DVector<f64>>::new(),
        StepKind::Block { .. }
    );
    check_inner_trace!(
        x,
        Binomial::new(&binary),
        Slope::new(vec![0.01, 0.005]),
        Hybrid::<DVector<f64>>::new(),
        StepKind::ProximalGradient
    );
}
