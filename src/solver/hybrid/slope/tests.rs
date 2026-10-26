use super::*;
use crate::solver::hybrid::tests::matrix::Matrix;
use crate::solver::hybrid::{Hybrid, HybridStep, Workspace, iterate};
use crate::solver::proximal::NativeDesign;
use crate::{Gaussian, Normalization, Problem, Termination};

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-10 * (1.0 + expected.abs()),
        "{actual} != {expected}"
    );
}

fn penalty_value(beta: &[f64], weights: &[f64]) -> f64 {
    let mut sorted: Vec<_> = beta.iter().map(|x| x.abs()).collect();
    sorted.sort_unstable_by(|a, b| b.total_cmp(a));
    sorted.iter().zip(weights).map(|(x, w)| x * w).sum()
}

// Enumerate signed intervals and sort all individual coefficient copies for
// each derivative, independently of the solver's incremental rank windows.
fn independent_scalar(
    q: f64,
    b: f64,
    current: f64,
    mass: usize,
    others: &[(f64, usize)],
    weights: &[f64],
) -> f64 {
    let penalty = |z: f64| {
        let mut beta = vec![z; mass];
        for &(value, count) in others {
            beta.extend(std::iter::repeat_n(value, count));
        }
        beta.resize(weights.len(), 0.0);
        penalty_value(&beta, weights)
    };
    let mut knots = vec![0.0];
    for &(value, _) in others {
        knots.extend([-value, value]);
    }
    knots.sort_unstable_by(f64::total_cmp);
    knots.dedup();
    let mut candidates = knots.clone();
    candidates.push(current);
    if q > 0.0 {
        for interval in 0..=knots.len() {
            let low = if interval == 0 {
                f64::NEG_INFINITY
            } else {
                knots[interval - 1]
            };
            let high = if interval == knots.len() {
                f64::INFINITY
            } else {
                knots[interval]
            };
            let midpoint = if !low.is_finite() {
                high - 1.0
            } else if !high.is_finite() {
                low + 1.0
            } else {
                0.5 * low + 0.5 * high
            };
            let mut entries = vec![(midpoint.abs(), true); mass];
            for &(value, count) in others {
                entries.extend(std::iter::repeat_n((value, false), count));
            }
            entries.resize(weights.len(), (0.0, false));
            entries.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
            let slope: f64 = entries
                .iter()
                .zip(weights)
                .filter(|((_, moving), _)| *moving)
                .map(|(_, weight)| weight * midpoint.signum())
                .sum();
            let root = (b - slope) / q;
            if root >= low && root <= high {
                return root;
            }
        }
        return candidates
            .into_iter()
            .min_by(|left, right| {
                let objective = |z: f64| 0.5 * q * z * z - b * z + penalty(z);
                objective(*left).total_cmp(&objective(*right))
            })
            .unwrap();
    }
    let objective = |z: f64| 0.5 * q * z * z - b * z + penalty(z);
    let best = candidates
        .iter()
        .map(|&z| objective(z))
        .fold(f64::INFINITY, f64::min);
    candidates
        .into_iter()
        .filter(|&z| (objective(z) - best).abs() <= 1e-12)
        .min_by(|a, b| (a - current).abs().total_cmp(&(b - current).abs()))
        .unwrap()
}

fn residual(rows: &[Vec<f64>], y: &[f64], beta: &[f64], intercept: f64) -> Vec<f64> {
    rows.iter()
        .zip(y)
        .map(|(row, y)| y - intercept - row.iter().zip(beta).map(|(x, b)| x * b).sum::<f64>())
        .collect()
}

fn objective(rows: &[Vec<f64>], y: &[f64], beta: &[f64], intercept: f64, weights: &[f64]) -> f64 {
    residual(rows, y, beta, intercept)
        .iter()
        .map(|r| r * r / (2.0 * y.len() as f64))
        .sum::<f64>()
        + penalty_value(beta, weights)
}

fn assert_state(state: &State, rows: &[Vec<f64>], y: &[f64], beta: &[f64], intercept: f64) {
    assert!(
        state
            .blocks
            .windows(2)
            .all(|pair| pair[0].value > pair[1].value)
    );
    let mut seen = vec![false; beta.len()];
    let mut prediction = vec![intercept; y.len()];
    for (label, block) in state.blocks.iter().enumerate() {
        assert!(!block.value.is_sign_negative());
        assert_eq!(state.positions[block.id], label);
        assert!(block.members.windows(2).all(|pair| pair[0] < pair[1]));
        for &j in &block.members {
            assert!(!seen[j]);
            seen[j] = true;
            assert_eq!(state.labels[j], label);
            assert_eq!(beta[j], state.signs[j] * block.value);
            assert_eq!(
                state.signs[j],
                if beta[j] == 0.0 {
                    0.0
                } else {
                    beta[j].signum()
                }
            );
        }
        let column: Vec<f64> = rows
            .iter()
            .map(|row| block.members.iter().map(|&j| row[j] * state.signs[j]).sum())
            .collect();
        for ((&actual, &expected), value) in block.column.iter().zip(&column).zip(&mut prediction) {
            close(actual, expected);
            *value += actual * block.value;
        }
        close(
            block.curvature,
            column.iter().map(|a| a * a / y.len() as f64).sum(),
        );
    }
    assert!(seen.iter().all(|x| *x));
    for (((actual, y), prediction), expected) in state
        .residual
        .iter()
        .zip(y)
        .zip(prediction)
        .zip(residual(rows, y, beta, intercept))
    {
        close(*actual, expected);
        close(y - prediction, expected);
    }
}

#[test]
fn magnitude_order_signs_zeros_and_permutations_reconstruct_predictions() {
    let adjacent = f64::from_bits(1.0_f64.to_bits() + 1);
    for beta in [
        vec![0.0, -2.0, 1.0, 2.0, -0.0, -1.0, adjacent],
        vec![adjacent, -1.0, -0.0, 2.0, 1.0, -2.0, 0.0],
    ] {
        let rows = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            vec![-1.0, 0.0, 2.0, 1.0, 0.0, -3.0, 1.0],
        ];
        let x = Matrix::from_rows(&rows.iter().map(Vec::as_slice).collect::<Vec<_>>());
        let y = [2.0, -1.0];
        let penalty = Slope::new(vec![2.0, 2.0, 1.0, 1.0, 0.0, 0.0, 0.0]);
        let mut state = State::new(2, 7);
        state
            .rebuild(
                &LazyMatrix::from_parts(&x, None, None),
                &beta,
                &residual(&rows, &y, &beta, 0.5),
                &penalty,
                0,
            )
            .unwrap();
        assert_eq!(state.values(), [2.0, adjacent, 1.0, 0.0]);
        assert_state(&state, &rows, &y, &beta, 0.5);
    }
}

#[test]
fn scalar_updates_match_independent_signed_interval_enumeration() {
    for others in [
        vec![],
        vec![(3.0, 1), (1.0, 2)],
        vec![(4.0, 2), (2.0, 1), (0.5, 1)],
    ] {
        for mass in 1..=3 {
            let p = mass + others.iter().map(|(_, n)| n).sum::<usize>() + 2;
            for weights in [
                vec![0.0; p],
                vec![0.25; p],
                (0..p).map(|j| ((p - j) / 2) as f64 / 4.0).collect(),
            ] {
                for q in [0.0, 0.25, 2.0] {
                    for b in [-10.0, -2.0, -0.25, 0.0, 0.25, 2.0, 10.0] {
                        if q == 0.0 && b != 0.0 {
                            continue;
                        }
                        for current in [0.125, 1.0, 5.0] {
                            let expected =
                                independent_scalar(q, b, current, mass, &others, &weights);
                            let actual =
                                scalar_minimum(q, b, current, mass, &others, &weights, 0).unwrap();
                            close(actual, expected);
                        }
                    }
                }
            }
        }
    }
    close(
        scalar_minimum(1.0, 1.0, 0.5, 1, &[(2.0, 1)], &[1e100, 0.25], 0).unwrap(),
        0.75,
    );
    close(
        scalar_minimum(1e-20, 2e-20, 1.0, 1, &[], &[0.0], 0).unwrap(),
        2.0,
    );
    assert!(scalar_minimum(0.0, 1.0, 1.0, 1, &[], &[0.0], 7).is_err());
}

fn reference_sweep(
    rows: &[Vec<f64>],
    y: &[f64],
    beta: &mut [f64],
    intercept: &mut f64,
    free_intercept: bool,
    penalty: &Slope,
) {
    let mut order: Vec<_> = (0..beta.len()).collect();
    order.sort_unstable_by(|&a, &b| beta[b].abs().total_cmp(&beta[a].abs()).then(a.cmp(&b)));
    let mut blocks: Vec<(usize, Vec<usize>)> = Vec::new();
    for j in order {
        if blocks
            .last()
            .is_none_or(|(_, members)| beta[members[0]].abs() != beta[j].abs())
        {
            blocks.push((j, vec![j]));
        } else {
            blocks.last_mut().unwrap().1.push(j);
        }
    }
    let visits: Vec<_> = blocks
        .iter()
        .filter(|(_, members)| beta[members[0]] != 0.0)
        .map(|(id, _)| *id)
        .collect();
    for id in visits {
        let Some(index) = blocks.iter().position(|(block_id, _)| *block_id == id) else {
            continue;
        };
        let members = &blocks[index].1;
        let current = beta[members[0]].abs();
        if current == 0.0 {
            continue;
        }
        let r = residual(rows, y, beta, *intercept);
        let column: Vec<f64> = rows
            .iter()
            .map(|row| members.iter().map(|&j| row[j] * beta[j].signum()).sum())
            .collect();
        let q = column.iter().map(|a| a * a / y.len() as f64).sum();
        let b = column
            .iter()
            .zip(r)
            .map(|(a, r)| a * (r + a * current) / y.len() as f64)
            .sum();
        let others: Vec<_> = blocks
            .iter()
            .enumerate()
            .filter(|(i, (_, members))| *i != index && beta[members[0]] != 0.0)
            .map(|(_, (_, members))| (beta[members[0]].abs(), members.len()))
            .collect();
        let value = independent_scalar(q, b, current, members.len(), &others, penalty.weights());
        for &j in members {
            beta[j] = if value == 0.0 {
                0.0
            } else {
                beta[j].signum() * value
            };
        }
        let mut moved = blocks.remove(index);
        if let Some(other) = blocks
            .iter()
            .position(|(_, members)| beta[members[0]].abs() == value.abs())
        {
            moved.1.extend(blocks.remove(other).1);
        }
        let position = blocks.partition_point(|(_, members)| beta[members[0]].abs() > value.abs());
        blocks.insert(position, moved);
    }
    if free_intercept {
        *intercept += residual(rows, y, beta, *intercept).iter().sum::<f64>() / y.len() as f64;
    }
}

#[test]
fn sweeps_match_rebuilding_every_restriction_and_preserve_state() {
    for seed in 0..12 {
        let rows: Vec<Vec<_>> = (0..6)
            .map(|i| {
                (0..7)
                    .map(|j| ((i * 13 + j * 7 + i * j + seed * 11) % 23) as f64 / 5.0 - 2.0)
                    .collect()
            })
            .collect();
        let x = Matrix::from_rows(&rows.iter().map(Vec::as_slice).collect::<Vec<_>>());
        let y = [2.0, -1.0, 3.0, 0.5, -2.0, 1.0];
        for weights in [
            vec![0.0; 7],
            vec![0.2; 7],
            vec![1.0, 0.75, 0.5, 0.25, 0.25, 0.0, 0.0],
            vec![100.0; 7],
        ] {
            for free_intercept in [false, true] {
                let penalty = Slope::new(weights.clone());
                let mut beta = [-1.0, 1.0, -2.0, 0.0, 2.0, 2.0, -0.5];
                let mut expected = beta;
                let (mut intercept, mut expected_intercept) = (0.0, 0.0);
                let mut state = State::new(6, 7);
                state
                    .rebuild(
                        &LazyMatrix::from_parts(&x, None, None),
                        &beta,
                        &residual(&rows, &y, &beta, intercept),
                        &penalty,
                        0,
                    )
                    .unwrap();
                for iteration in 1..=5 {
                    let before = objective(&rows, &y, &beta, intercept, penalty.weights());
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
                    for (&actual, &expected) in beta.iter().zip(&expected) {
                        close(actual, expected);
                    }
                    close(intercept, expected_intercept);
                    assert!(
                        objective(&rows, &y, &beta, intercept, penalty.weights())
                            <= before + 1e-10 * (1.0 + before)
                    );
                    assert_eq!(beta[3], 0.0);
                    assert_state(&state, &rows, &y, &beta, intercept);
                }
            }
        }
    }
}

#[test]
fn sign_flip_merges_recompute_cross_terms_and_retire_absorbed_ids() {
    let rows = vec![vec![1.0, 2.0, 3.0]];
    let x = Matrix::from_rows(&[&rows[0]]);
    let y = [-5.0];
    let penalty = Slope::new(vec![3.0, 1.0, 0.0]);
    let mut beta = [3.0, -1.0, 0.0];
    let mut state = State::new(1, 3);
    state
        .rebuild(
            &LazyMatrix::from_parts(&x, None, None),
            &beta,
            &residual(&rows, &y, &beta, 0.0),
            &penalty,
            0,
        )
        .unwrap();
    state
        .sweep::<std::convert::Infallible>(&mut beta, &mut 0.0, false, &penalty, 1)
        .unwrap();
    assert_eq!(beta, [-1.0, -1.0, 0.0]);
    assert_eq!(state.blocks[0].column, [-3.0]);
    assert_eq!(state.blocks[0].curvature, 9.0);
    assert_eq!(state.positions, [0, usize::MAX, 1]);
    assert_state(&state, &rows, &y, &beta, 0.0);
}

#[test]
fn zero_curvature_projects_onto_flat_minima_and_does_not_reactivate_zeros() {
    let x = Matrix::empty(2, 3);
    let matrix = LazyMatrix::from_parts(&x, None, None);
    for (weights, expected) in [
        (vec![0.0; 3], [3.0, -1.0, 0.0]),
        (vec![1.0, 0.0, 0.0], [0.0; 3]),
    ] {
        let penalty = Slope::new(weights);
        let mut beta = [3.0, -1.0, 0.0];
        let mut state = State::new(2, 3);
        state
            .rebuild(&matrix, &beta, &[1.0, -1.0], &penalty, 0)
            .unwrap();
        state
            .sweep::<std::convert::Infallible>(&mut beta, &mut 0.0, false, &penalty, 1)
            .unwrap();
        if expected == [0.0; 3] {
            assert_eq!(beta, [1.0, -1.0, 0.0]);
        }
        state
            .sweep::<std::convert::Infallible>(&mut beta, &mut 0.0, false, &penalty, 2)
            .unwrap();
        assert_eq!(beta, expected);
        assert_eq!(state.residual, [1.0, -1.0]);
    }
    let tiny = Matrix::from_rows(&[&[1e-200]]);
    assert!(
        State::new(1, 1)
            .rebuild(
                &LazyMatrix::from_parts(&tiny, None, None),
                &[1.0],
                &[0.0],
                &Slope::new(vec![0.0]),
                0
            )
            .is_err()
    );
}

#[test]
fn opposite_signs_can_cancel_an_aggregate_without_zero_design_columns() {
    let rows = vec![vec![1.0, 1.0, 2.0], vec![2.0, 2.0, -1.0]];
    let x = Matrix::from_rows(&rows.iter().map(Vec::as_slice).collect::<Vec<_>>());
    let y = [1.0, -1.0];
    let penalty = Slope::new(vec![0.5, 0.1, 0.0]);
    let mut beta = [1.0, -1.0, 0.0];
    let mut state = State::new(2, 3);
    state
        .rebuild(
            &LazyMatrix::from_parts(&x, None, None),
            &beta,
            &y,
            &penalty,
            0,
        )
        .unwrap();
    assert_eq!(state.blocks[0].column, [0.0; 2]);
    assert_eq!(state.blocks[0].curvature, 0.0);
    state
        .sweep::<std::convert::Infallible>(&mut beta, &mut 0.0, false, &penalty, 1)
        .unwrap();
    assert_eq!(beta, [0.0; 3]);
    assert_eq!(state.residual, y);
    assert_state(&state, &rows, &y, &beta, 0.0);
}

#[test]
fn rebuilds_reuse_storage_when_signs_and_membership_change() {
    let x = Matrix::from_rows(&[&[1.0, 2.0, 3.0], &[2.0, -1.0, 1.0]]);
    let matrix = LazyMatrix::from_parts(&x, None, None);
    let penalty = Slope::new(vec![1.0; 3]);
    let mut state = State::new(2, 3);
    state
        .rebuild(&matrix, &[1.0, -1.0, 2.0], &[0.0; 2], &penalty, 0)
        .unwrap();
    let mut pointers: Vec<_> = state.blocks.iter().map(|b| b.column.as_ptr()).collect();
    state
        .rebuild(&matrix, &[-1.0, -2.0, 2.0], &[1.0; 2], &penalty, 1)
        .unwrap();
    let mut rebuilt: Vec<_> = state.blocks.iter().map(|b| b.column.as_ptr()).collect();
    pointers.sort();
    rebuilt.sort();
    assert_eq!(pointers, rebuilt);
    assert_eq!(state.labels, [1, 0, 0]);
    assert_eq!(state.blocks[0].column, [1.0, 2.0]);
    assert_eq!(state.blocks[1].column, [-1.0, -2.0]);
}

#[test]
fn full_pg_splits_magnitude_clusters_and_reactivates_the_zero_block() {
    let x = Matrix::from_rows(&[&[1.0, 0.0, 0.0], &[0.0, 2.0, 0.0], &[0.0, 0.0, 1.0]]);
    let problem = Problem::new(
        &x,
        Gaussian::new(&[4.0, -1.0, 3.0]).fit_intercept(false),
        Slope::new(vec![0.2, 0.1, 0.0]),
    )
    .normalize(Normalization::None);
    for check_every in [1, 4] {
        let mut design = NativeDesign::<_, Vec<f64>>::new(problem.prepare(false).unwrap());
        let mut work = Workspace::<Slope>::new(3, 3);
        work.coefficients.copy_from_slice(&[1.0, -1.0, 0.0]);
        let options = Hybrid::<Vec<f64>>::new()
            .max_iterations(7)
            .check_every(check_every)
            .step_policy(HybridStep::FixedBound);
        let solution = iterate(
            &mut design,
            *problem.datafit(),
            problem.penalty(),
            &options,
            1e-14,
            &mut work,
        )
        .unwrap();
        assert_eq!(solution.pg_steps, 2);
        assert_eq!(solution.coordinate_sweeps, 5);
        assert_eq!(solution.proximal.termination, Termination::IterationLimit);
        assert_ne!(solution.labels[0], solution.labels[1]);
        assert_ne!(solution.proximal.coefficients[2], 0.0);
    }
}

#[test]
fn unchanged_sweeps_wait_for_pg_to_reactivate_a_zero_coordinate() {
    let x = Matrix::from_rows(&[&[1.0, 1.0], &[0.0, 1.0]]);
    let problem = Problem::new(
        &x,
        Gaussian::new(&[2.0, -1.0]).fit_intercept(false),
        Slope::new(vec![0.1, 0.0]),
    )
    .normalize(Normalization::None);
    for check_every in [1, 4] {
        for budget in [6, 7] {
            let mut design = NativeDesign::<_, Vec<f64>>::new(problem.prepare(false).unwrap());
            let mut work = Workspace::<Slope>::new(2, 2);
            work.coefficients.copy_from_slice(&[1.0, 0.0]);
            let options = Hybrid::<Vec<f64>>::new()
                .max_iterations(budget)
                .check_every(check_every)
                .step_policy(HybridStep::FixedBound);
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
            assert_eq!(solution.proximal.coefficients[1] == 0.0, budget == 6);
            assert_eq!(solution.proximal.termination, Termination::IterationLimit);
            assert!(solution.proximal.stopping_value > 0.1);
        }
    }
}
