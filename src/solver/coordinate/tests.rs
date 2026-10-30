use std::cell::Cell;

use lazymatrix::MatrixShape;

use super::*;

struct Sparse {
    nrows: usize,
    columns: Vec<Vec<(usize, f64)>>,
    visits: Cell<usize>,
}

struct Column<'a> {
    matrix: &'a Sparse,
    j: usize,
}

impl MatrixShape for Sparse {
    fn nrows(&self) -> usize {
        self.nrows
    }
    fn ncols(&self) -> usize {
        self.columns.len()
    }
}

impl RawColumns<f64> for Sparse {
    type Column<'a> = Column<'a>;
    fn raw_column(&self, j: usize) -> Self::Column<'_> {
        Column { matrix: self, j }
    }
}

impl RawColumn<f64> for Column<'_> {
    fn len(&self) -> usize {
        self.matrix.nrows
    }
    fn stored_len(&self) -> usize {
        self.matrix.columns[self.j].len()
    }
    fn for_each_stored(&self, mut f: impl FnMut(usize, f64)) {
        for &(i, value) in &self.matrix.columns[self.j] {
            self.matrix.visits.set(self.matrix.visits.get() + 1);
            f(i, value);
        }
    }
}

fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-10 * (1.0 + b.abs()), "{a} != {b}");
}

#[test]
fn coordinate_update_and_cached_dot_visit_only_stored_entries() {
    for nrows in [10, 100_000] {
        let raw = Sparse {
            nrows,
            columns: vec![vec![(1, 2.0), (5, -3.0), (8, 0.0)]],
            visits: Cell::new(0),
        };
        let matrix = LazyMatrix::from_parts(&raw, Some(vec![0.25]), Some(vec![2.0]));
        let column = matrix.column(0);
        let summary = ColumnSummary::new(&column).unwrap();
        let mut residual = Residual::new(&vec![1.0; nrows], 0.0).unwrap();
        raw.visits.set(0);
        residual.subtract_column(&column, &summary, 0.5, 1).unwrap();
        assert_eq!(raw.visits.get(), 3);
        close(residual.base[0], 1.0);
        close(residual.base[1], 0.5);
        close(residual.base[5], 1.75);
        close(residual.offset, 0.0625);
        raw.visits.set(0);
        let dot = residual.dot(&column, &summary, 1).unwrap();
        assert_eq!(raw.visits.get(), 3);
        let expected: f64 = (0..nrows)
            .map(|i| {
                let raw = match i {
                    1 => 2.0,
                    5 => -3.0,
                    _ => 0.0,
                };
                let x = (raw - 0.25) / 2.0;
                x * (1.0 - 0.5 * x)
            })
            .sum();
        close(dot, expected);
    }
}

#[test]
fn repeated_updates_and_refresh_match_materialized_residuals() {
    let raw = Sparse {
        nrows: 5,
        columns: vec![vec![(0, 2.0), (3, -1.0)], vec![(1, 4.0), (4, 3.0)]],
        visits: Cell::new(0),
    };
    let matrix = LazyMatrix::from_parts(&raw, Some(vec![0.2, 1.4]), Some(vec![0.8, 2.0]));
    let summaries: Vec<_> = (0..2)
        .map(|j| ColumnSummary::new(&matrix.column(j)).unwrap())
        .collect();
    let explicit = [
        [2.25, -0.7],
        [-0.25, 1.3],
        [-0.25, -0.7],
        [-1.5, -0.7],
        [-0.25, 0.8],
    ];
    let y = [2.0, -1.0, 4.0, 0.0, 3.0];
    let mut coefficients = [0.0; 2];
    let mut intercept = 0.5;
    let mut residual = Residual::new(&y, intercept).unwrap();
    for iteration in 1..=250 {
        let j = iteration % 2;
        let delta = if iteration % 3 == 0 { -0.023 } else { 0.017 };
        coefficients[j] += delta;
        residual
            .subtract_column(&matrix.column(j), &summaries[j], delta, iteration)
            .unwrap();
        intercept += 0.003;
        residual.offset -= 0.003;
        if iteration % CHECK_INTERVAL == 0 {
            residual
                .refresh(&matrix, &y, &coefficients, intercept, &summaries, iteration)
                .unwrap();
        }
        let reference: Vec<_> = explicit
            .iter()
            .zip(y)
            .map(|(row, y)| y - intercept - row[0] * coefficients[0] - row[1] * coefficients[1])
            .collect();
        for (&base, &expected) in residual.base.iter().zip(&reference) {
            close(base + residual.offset, expected);
        }
        close(residual.base_sum, residual.base.iter().sum());
        close(
            residual.mean(iteration).unwrap(),
            reference.iter().sum::<f64>() / 5.0,
        );
        close(
            residual.loss(iteration).unwrap(),
            reference.iter().map(|r| r * r).sum::<f64>() / 10.0,
        );
        for j in 0..2 {
            close(
                residual
                    .dot(&matrix.column(j), &summaries[j], iteration)
                    .unwrap(),
                explicit
                    .iter()
                    .zip(&reference)
                    .map(|(row, r)| row[j] * r)
                    .sum(),
            );
        }
    }
    // A refresh must discard accumulated cache drift, including drift in the
    // scalar offset, rather than only resumming an already stale residual.
    residual.base_sum += 1.0;
    residual.offset += 0.1;
    residual.base[0] += 0.2;
    residual
        .refresh(&matrix, &y, &coefficients, intercept, &summaries, 250)
        .unwrap();
    for (i, row) in explicit.iter().enumerate() {
        close(
            residual.base[i] + residual.offset,
            y[i] - intercept - row[0] * coefficients[0] - row[1] * coefficients[1],
        );
    }
}

#[test]
fn path_workspace_retains_storage_and_rebuilds_screening_at_each_strength() {
    let raw = Sparse {
        nrows: 4,
        columns: vec![
            vec![(0, -1.0), (1, -1.0), (2, 1.0), (3, 1.0)],
            vec![(0, -1.0), (1, 1.0), (2, -1.0), (3, 1.0)],
        ],
        visits: Cell::new(0),
    };
    let matrix = LazyMatrix::from_parts(&raw, None, None);
    let y = [-3.1, -2.9, 2.9, 3.1];
    let datafit = Gaussian::new(&y).fit_intercept(false);
    let options = CoordinateDescent::new().tolerance(1e-10);
    let mut work = Workspace::new(&matrix, datafit).unwrap();
    let pointers = (
        work.coefficients.as_ptr(),
        work.residual.base.as_ptr(),
        work.summaries.as_ptr(),
        work.dual_certificate.as_ptr(),
        work.dual_correlations.as_ptr(),
        work.correlations.as_ptr(),
        work.working.selected.as_ptr(),
        work.screened.as_ptr(),
    );
    let first = solve_with_workspace(&matrix, datafit, L1::new(1.0), &options, &mut work).unwrap();
    assert_eq!(first.termination, Termination::Converged);
    assert!(work.screened[1]);
    assert!(!work.working.selected[1]);
    let repeated =
        solve_with_workspace(&matrix, datafit, L1::new(1.0), &options, &mut work).unwrap();
    assert_eq!(repeated.iterations, 0);
    let weaker =
        solve_with_workspace(&matrix, datafit, L1::new(0.01), &options, &mut work).unwrap();
    assert_eq!(weaker.termination, Termination::Converged);
    close(weaker.coefficients[1], 0.09);
    assert!(!work.screened[1]);
    assert!(work.working.selected[1]);
    assert_eq!(
        pointers,
        (
            work.coefficients.as_ptr(),
            work.residual.base.as_ptr(),
            work.summaries.as_ptr(),
            work.dual_certificate.as_ptr(),
            work.dual_correlations.as_ptr(),
            work.correlations.as_ptr(),
            work.working.selected.as_ptr(),
            work.screened.as_ptr(),
        )
    );
    let zero = solve_with_workspace(&matrix, datafit, L1::new(0.0), &options, &mut work).unwrap();
    assert_eq!(zero.termination, Termination::Converged);
    assert!(work.screened.iter().all(|screened| !screened));
}

#[test]
fn path_iteration_limits_keep_optimization_parameters() {
    let raw = Sparse {
        nrows: 3,
        columns: vec![vec![(0, 1.0), (1, 2.0), (2, 3.0)]],
        visits: Cell::new(0),
    };
    let matrix = LazyMatrix::from_parts(&raw, None, None);
    let datafit = Gaussian::new(&[2.0, 4.0, 8.0]);
    let options = CoordinateDescent::new().tolerance(1e-12).max_iterations(1);
    let mut work = Workspace::new(&matrix, datafit).unwrap();
    let first = solve_with_workspace(&matrix, datafit, L1::new(0.1), &options, &mut work).unwrap();
    let second = solve_with_workspace(&matrix, datafit, L1::new(0.1), &options, &mut work).unwrap();
    assert_eq!(first.termination, Termination::IterationLimit);
    assert_eq!(second.termination, Termination::IterationLimit);
    assert_eq!(first.iterations, 1);
    assert_eq!(second.iterations, 1);
    assert!(second.objective < first.objective);
    assert!(second.coefficients[0] > first.coefficients[0]);
}

#[test]
fn full_checks_wait_for_ten_sweeps_even_when_coordinates_stop_moving() {
    let raw = Sparse {
        nrows: 2,
        columns: vec![vec![(0, -1.0), (1, 1.0)]],
        visits: Cell::new(0),
    };
    let matrix = LazyMatrix::from_parts(&raw, None, None);
    for policy in [
        crate::WorkingSetPolicy::Disabled,
        crate::WorkingSetPolicy::EverActive,
        crate::WorkingSetPolicy::Auto,
    ] {
        for criterion in [
            StoppingCriterion::kkt_violation(1e-10),
            StoppingCriterion::duality_gap(1e-10),
        ] {
            let options = CoordinateDescent::new()
                .working_set_policy(policy)
                .terminate_on(criterion);
            let fit = solve(
                &matrix,
                Gaussian::new(&[-2.0, 2.0]).fit_intercept(false),
                L1::new(0.5),
                &options,
            )
            .unwrap();
            assert_eq!(fit.termination, Termination::Converged);
            assert_eq!(fit.iterations, 10, "{policy:?}, {criterion:?}");
            assert_eq!(fit.working_set.full_checks, 2);
            assert_eq!(fit.working_set.restricted_checks, 0);
            close(fit.coefficients[0], 1.5);
        }
    }
}

#[test]
fn full_checks_include_iteration_limits_between_ten_sweep_boundaries() {
    let raw = Sparse {
        nrows: 2,
        columns: vec![vec![(0, 1.0), (1, 1.0)], vec![(0, 1.0), (1, 1.01)]],
        visits: Cell::new(0),
    };
    let matrix = LazyMatrix::from_parts(&raw, None, None);
    for limit in [1, 7, 10, 11, 23] {
        let fit = solve(
            &matrix,
            Gaussian::new(&[-1.0, 1.0]).fit_intercept(false),
            L1::new(0.0),
            &CoordinateDescent::new()
                .working_sets(false)
                .tolerance(1e-14)
                .max_iterations(limit),
        )
        .unwrap();
        assert_eq!(fit.termination, Termination::IterationLimit);
        assert_eq!(fit.iterations, limit);
        assert_eq!(fit.working_set.full_checks, 1 + limit.div_ceil(10));
        assert_eq!(fit.working_set.restricted_checks, 0);
        let mut work =
            Workspace::new(&matrix, Gaussian::new(&[-1.0, 1.0]).fit_intercept(false)).unwrap();
        work.coefficients.copy_from_slice(&fit.coefficients);
        let final_check = work
            .diagnose(
                &matrix,
                Gaussian::new(&[-1.0, 1.0]).fit_intercept(false),
                L1::new(0.0),
                limit,
            )
            .unwrap();
        close(fit.kkt_violation, final_check.kkt_violation);
        close(fit.objective, final_check.objective);
    }
}

#[test]
fn scheduled_full_checks_expand_the_working_set_until_all_violations_are_resolved() {
    let raw = Sparse {
        nrows: 24,
        columns: (0..24).map(|j| vec![(j, 1.0)]).collect(),
        visits: Cell::new(0),
    };
    let matrix = LazyMatrix::from_parts(&raw, None, None);
    let fit = solve(
        &matrix,
        Gaussian::new(&[1.0; 24]).fit_intercept(false),
        L1::new(0.01),
        &CoordinateDescent::new().tolerance(1e-12),
    )
    .unwrap();
    assert_eq!(fit.termination, Termination::Converged);
    assert_eq!(fit.iterations, 30);
    assert_eq!(fit.working_set.full_checks, 4);
    assert_eq!(fit.working_set.restricted_checks, 0);
    assert_eq!(fit.working_set.peak_active_dimension, 24);
    for coefficient in fit.coefficients {
        close(coefficient, 0.76);
    }
}
