use super::*;
use crate::solver::hybrid::{Hybrid, HybridStep, Workspace, iterate, tests::independent_scalar};
use crate::solver::proximal::NativeDesign;
use crate::{Gaussian, Normalization, Problem, Termination};

use crate::solver::hybrid::tests::matrix::Matrix;

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 3e-10 * (1.0 + expected.abs()),
        "{actual} != {expected}"
    );
}

fn residual(rows: &[Vec<f64>], y: &[f64], beta: &[f64], intercept: f64) -> Vec<f64> {
    rows.iter()
        .zip(y)
        .map(|(row, y)| y - intercept - row.iter().zip(beta).map(|(x, b)| x * b).sum::<f64>())
        .collect()
}

fn objective(
    rows: &[Vec<f64>],
    y: &[f64],
    beta: &[f64],
    intercept: f64,
    penalty: &ChainFusedLasso,
) -> f64 {
    residual(rows, y, beta, intercept)
        .iter()
        .map(|r| r * r / (2.0 * y.len() as f64))
        .sum::<f64>()
        + penalty.l1() * beta.iter().map(|b| b.abs()).sum::<f64>()
        + penalty.fusion()
            * beta
                .windows(2)
                .enumerate()
                .map(|(j, pair)| penalty.weight(j) * (pair[1] - pair[0]).abs())
                .sum::<f64>()
}

fn assert_state(
    state: &State,
    rows: &[Vec<f64>],
    y: &[f64],
    beta: &[f64],
    intercept: f64,
    penalty: &ChainFusedLasso,
) {
    let mut end = 0;
    for (label, segment) in state.segments.iter().enumerate() {
        assert_eq!(segment.start, end);
        assert!(segment.end > segment.start);
        assert_eq!(state.positions[segment.id], label);
        assert!(segment.value != 0.0 || segment.value.is_sign_positive());
        assert!(
            beta[segment.start..segment.end]
                .iter()
                .all(|&b| b == segment.value)
        );
        assert!(
            state.labels[segment.start..segment.end]
                .iter()
                .all(|&l| l == label)
        );
        let column: Vec<f64> = rows
            .iter()
            .map(|row| row[segment.start..segment.end].iter().sum())
            .collect();
        for (&a, &b) in segment.column.iter().zip(&column) {
            close(a, b);
        }
        close(
            segment.curvature,
            column.iter().map(|a| a * a / rows.len() as f64).sum(),
        );
        assert_eq!(
            segment.left_weight,
            if segment.start == 0 {
                0.0
            } else {
                penalty.weight(segment.start - 1)
            }
        );
        assert_eq!(
            segment.right_weight,
            if segment.end == beta.len() {
                0.0
            } else {
                penalty.weight(segment.end - 1)
            }
        );
        if label > 0 {
            assert!(segment.left_weight == 0.0 || state.segments[label - 1].value != segment.value);
        }
        end = segment.end;
    }
    assert_eq!(end, beta.len());
    for (&a, b) in state
        .residual
        .iter()
        .zip(residual(rows, y, beta, intercept))
    {
        close(a, b);
    }
}

#[test]
fn feature_order_cuts_signed_zero_and_unequal_adjacent_floats() {
    let rows = vec![
        vec![1.0, -1.0, 2.0, 3.0, -2.0, 1.0, 4.0],
        vec![2.0, -2.0, -1.0, 1.0, 2.0, 3.0, 2.0],
    ];
    let x = Matrix::from_rows(&rows.iter().map(Vec::as_slice).collect::<Vec<_>>());
    let matrix = LazyMatrix::from_parts(&x, None, None);
    let beta = [
        0.0,
        -0.0,
        1.0,
        1.0,
        0.0,
        1.0,
        f64::from_bits(1.0_f64.to_bits() + 1),
    ];
    let penalty = ChainFusedLasso::new(0.1, 0.3).with_weights(vec![1.0, 2.0, 0.0, 1.0, 0.5, 1.0]);
    let y = [1.0, -2.0];
    let mut state = State::new(2, beta.len());
    state
        .rebuild(
            &matrix,
            &beta,
            &residual(&rows, &y, &beta, 0.0),
            &penalty,
            0,
        )
        .unwrap();
    assert_eq!(state.labels, [0, 0, 1, 2, 3, 4, 5]);
    assert_eq!(state.segments[0].curvature, 0.0);
    assert_state(&state, &rows, &y, &beta, 0.0, &penalty);
    let empty = Matrix::empty(2, 0);
    state = State::new(2, 0);
    state
        .rebuild(
            &LazyMatrix::from_parts(&empty, None, None),
            &[],
            &y,
            &ChainFusedLasso::new(0.0, 0.0),
            0,
        )
        .unwrap();
    assert!(state.segments.is_empty());
}

fn reference_sweep(
    rows: &[Vec<f64>],
    y: &[f64],
    beta: &mut [f64],
    intercept: &mut f64,
    free_intercept: bool,
    penalty: &ChainFusedLasso,
) {
    let mut ranges: Vec<(usize, usize, usize)> = Vec::new();
    for j in 0..beta.len() {
        if j == 0 || beta[j] != beta[j - 1] || penalty.weight(j - 1) == 0.0 {
            ranges.push((j, j, j + 1));
        } else {
            ranges.last_mut().unwrap().2 = j + 1;
        }
    }
    let visits: Vec<_> = ranges.iter().map(|r| r.0).collect();
    for id in visits {
        let Some(mut index) = ranges.iter().position(|r| r.0 == id) else {
            continue;
        };
        let (_, start, end) = ranges[index];
        let column: Vec<f64> = rows
            .iter()
            .map(|row| row[start..end].iter().sum())
            .collect();
        let current = beta[start];
        let r = residual(rows, y, beta, *intercept);
        let q = column.iter().map(|a| a * a / y.len() as f64).sum();
        let h = column
            .iter()
            .zip(r)
            .map(|(a, r)| a * (r + a * current) / y.len() as f64)
            .sum();
        let mut knots = vec![(0.0, penalty.l1() * (end - start) as f64)];
        if start > 0 {
            knots.push((
                beta[start - 1],
                penalty.fusion() * penalty.weight(start - 1),
            ));
        }
        if end < beta.len() {
            knots.push((beta[end], penalty.fusion() * penalty.weight(end - 1)));
        }
        let value = independent_scalar(q, h, current, &knots);
        beta[start..end].fill(value);
        let mut range = ranges.remove(index);
        if index > 0 && penalty.weight(range.1 - 1) > 0.0 && beta[range.1 - 1] == value {
            range.1 = ranges.remove(index - 1).1;
            index -= 1;
        }
        if index < ranges.len() && penalty.weight(range.2 - 1) > 0.0 && beta[range.2] == value {
            range.2 = ranges.remove(index).2;
        }
        ranges.insert(index, range);
    }
    if free_intercept {
        *intercept += residual(rows, y, beta, *intercept).iter().sum::<f64>() / y.len() as f64;
    }
}

#[test]
fn sweeps_match_independently_rebuilt_restricted_problems() {
    for seed in 0..12 {
        let rows: Vec<Vec<_>> = (0..6)
            .map(|i| {
                (0..7)
                    .map(|j| ((i * 13 + j * 7 + i * j + seed * 11) % 23) as f64 / 5.0 - 2.0)
                    .collect()
            })
            .collect();
        let x = Matrix::from_rows(&rows.iter().map(Vec::as_slice).collect::<Vec<_>>());
        let matrix = LazyMatrix::from_parts(&x, None, None);
        let y = [2.0, -1.0, 3.0, 0.5, -2.0, 1.0];
        for (l1, fusion) in [(0.0, 0.0), (0.2, 0.0), (0.0, 0.3), (0.2, 0.3), (0.2, 100.0)] {
            for weights in [vec![1.0; 6], vec![0.5, 0.0, 2.0, 1.0, 0.25, 0.0]] {
                for free_intercept in [false, true] {
                    let penalty = ChainFusedLasso::new(l1, fusion).with_weights(weights.clone());
                    let mut beta = [-1.0, -1.0, 2.0, 0.0, 2.0, 2.0, -1.0];
                    let mut expected = beta;
                    let (mut intercept, mut expected_intercept) = (0.0, 0.0);
                    let mut state = State::new(y.len(), beta.len());
                    state
                        .rebuild(
                            &matrix,
                            &beta,
                            &residual(&rows, &y, &beta, intercept),
                            &penalty,
                            0,
                        )
                        .unwrap();
                    for iteration in 1..=5 {
                        let before = objective(&rows, &y, &beta, intercept, &penalty);
                        reference_sweep(
                            &rows,
                            &y,
                            &mut expected,
                            &mut expected_intercept,
                            free_intercept,
                            &penalty,
                        );
                        state
                            .sweep::<std::convert::Infallible>(
                                &mut beta,
                                &mut intercept,
                                free_intercept,
                                &penalty,
                                iteration,
                            )
                            .unwrap();
                        for (&a, &b) in beta.iter().zip(&expected) {
                            close(a, b);
                        }
                        close(intercept, expected_intercept);
                        assert!(
                            objective(&rows, &y, &beta, intercept, &penalty)
                                <= before + 1e-10 * (1.0 + before)
                        );
                        assert_state(&state, &rows, &y, &beta, intercept, &penalty);
                    }
                }
            }
        }
    }
}

#[test]
fn adjacent_merges_include_cross_terms_and_preserve_the_moving_identity() {
    let x = Matrix::from_rows(&[&[1.0, 2.0, 3.0]]);
    let penalty = ChainFusedLasso::new(0.0, 1.0);
    let mut state = State::new(1, 3);
    state
        .rebuild(
            &LazyMatrix::from_parts(&x, None, None),
            &[1.0, 0.0, 1.0],
            &[0.0],
            &penalty,
            0,
        )
        .unwrap();
    state
        .move_segment::<std::convert::Infallible>(1, 1.0, 1)
        .unwrap();
    assert_eq!(state.segments.len(), 1);
    assert_eq!(state.segments[0].id, 1);
    assert_eq!(state.segments[0].column, [6.0]);
    assert_eq!(state.segments[0].curvature, 36.0);
    assert_eq!(state.positions, [usize::MAX, 0, usize::MAX]);
    assert_eq!(state.labels, [0, 0, 0]);
}

#[test]
fn zero_curvature_uses_weighted_medians_and_positive_curvature_never_rounds_to_zero() {
    let x = Matrix::empty(2, 4);
    let penalty = ChainFusedLasso::new(0.2, 0.3).with_weights(vec![1.0, 0.0, 2.0]);
    let mut beta = [2.0, -1.0, 3.0, 3.0];
    let mut state = State::new(2, 4);
    state
        .rebuild(
            &LazyMatrix::from_parts(&x, None, None),
            &beta,
            &[1.0, -1.0],
            &penalty,
            0,
        )
        .unwrap();
    state
        .sweep::<std::convert::Infallible>(&mut beta, &mut 0.0, false, &penalty, 1)
        .unwrap();
    assert_eq!(beta, [-1.0, -1.0, 0.0, 0.0]);
    state
        .sweep::<std::convert::Infallible>(&mut beta, &mut 0.0, false, &penalty, 2)
        .unwrap();
    assert_eq!(beta, [0.0; 4]);
    assert_eq!(state.residual, [1.0, -1.0]);
    let tiny = Matrix::from_rows(&[&[1e-200]]);
    assert!(
        State::new(1, 1)
            .rebuild(
                &LazyMatrix::from_parts(&tiny, None, None),
                &[0.0],
                &[0.0],
                &ChainFusedLasso::new(0.0, 0.0),
                0
            )
            .is_err()
    );
}

#[test]
fn pg_splits_blocks_and_diagnostics_do_not_advance_the_sweep_schedule() {
    let x = Matrix::from_rows(&[&[1.0, 0.0], &[0.0, 1.0]]);
    let problem = Problem::new(
        &x,
        Gaussian::new(&[-2.0, 2.0]).fit_intercept(false),
        ChainFusedLasso::new(0.05, 0.1),
    )
    .normalize(Normalization::None);
    for check_every in [1, 3] {
        for budget in [6, 7] {
            let mut design = NativeDesign::<_, Vec<f64>>::new(problem.prepare(false).unwrap());
            let mut work = Workspace::<ChainFusedLasso>::new(2, 2);
            work.coefficients.copy_from_slice(&[2.0 / 3.0, -2.0 / 3.0]);
            let options = Hybrid::<Vec<f64>>::new()
                .max_iterations(budget)
                .check_every(check_every)
                .step_policy(HybridStep::Backtracking {
                    initial_curvature: Some(2.0),
                    max_backtracks: 100,
                });
            let solution = iterate(
                &mut design,
                *problem.datafit(),
                problem.penalty(),
                &options,
                1e-14,
                &mut work,
            )
            .unwrap();
            assert_eq!(solution.coordinate_sweeps, 5);
            assert_eq!(solution.pg_steps, if budget == 6 { 1 } else { 2 });
            assert_eq!(solution.labels[0] == solution.labels[1], budget == 6);
            assert_eq!(solution.proximal.termination, Termination::IterationLimit);
        }
    }
}

#[test]
fn rebuilds_reuse_columns_even_when_only_segment_boundaries_change() {
    let x = Matrix::from_rows(&[&[1.0, 2.0, 3.0], &[2.0, -1.0, 1.0]]);
    let matrix = LazyMatrix::from_parts(&x, None, None);
    let penalty = ChainFusedLasso::new(0.0, 1.0);
    let mut state = State::new(2, 3);
    state
        .rebuild(&matrix, &[1.0, 1.0, 2.0], &[0.0; 2], &penalty, 0)
        .unwrap();
    let mut pointers: Vec<_> = state.segments.iter().map(|s| s.column.as_ptr()).collect();
    state
        .rebuild(&matrix, &[1.0, 2.0, 2.0], &[1.0; 2], &penalty, 1)
        .unwrap();
    let mut rebuilt: Vec<_> = state.segments.iter().map(|s| s.column.as_ptr()).collect();
    pointers.sort();
    rebuilt.sort();
    assert_eq!(pointers, rebuilt);
    assert_eq!(state.labels, [0, 1, 1]);
    assert_eq!(state.segments[0].column, [1.0, 2.0]);
    assert_eq!(state.segments[1].column, [5.0, 0.0]);
}
