use super::*;
use lazymatrix::LazyMatrix;

#[path = "../../../tests/common/matrix.rs"]
mod matrix;
use matrix::Matrix;

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 2e-11 * (1.0 + expected.abs()),
        "{actual} != {expected}"
    );
}

fn residual(x: &[Vec<f64>], y: &[f64], beta: &[f64]) -> Vec<f64> {
    x.iter()
        .zip(y)
        .map(|(row, &y)| y - row.iter().zip(beta).map(|(x, b)| x * b).sum::<f64>())
        .collect()
}

fn objective(x: &[Vec<f64>], y: &[f64], beta: &[f64], l1: f64, fusion: f64) -> f64 {
    residual(x, y, beta)
        .iter()
        .map(|r| r * r / (2.0 * y.len() as f64))
        .sum::<f64>()
        + l1 * beta.iter().map(|b| b.abs()).sum::<f64>()
        + fusion
            * (0..beta.len())
                .flat_map(|i| (i + 1..beta.len()).map(move |j| (beta[i] - beta[j]).abs()))
                .sum::<f64>()
}

fn assert_state(state: &State, x: &[Vec<f64>], y: &[f64], beta: &[f64]) {
    for (j, &label) in state.labels.iter().enumerate() {
        assert_eq!(state.blocks[label].value, beta[j]);
        assert!(state.blocks[label].members.contains(&j));
    }
    assert!(
        state
            .blocks
            .windows(2)
            .all(|blocks| blocks[0].value < blocks[1].value)
    );
    let expected_residual = residual(x, y, beta);
    for (&actual, &expected) in state.residual.iter().zip(&expected_residual) {
        close(actual, expected);
    }
    let mut fit = vec![0.0; x.len()];
    for block in &state.blocks {
        assert!(block.members.windows(2).all(|pair| pair[0] < pair[1]));
        let column: Vec<_> = x
            .iter()
            .map(|row| block.members.iter().map(|&j| row[j]).sum::<f64>())
            .collect();
        for ((actual, &expected), predicted) in block.column.iter().zip(&column).zip(&mut fit) {
            close(*actual, expected);
            *predicted += actual * block.value;
        }
        close(
            block.curvature,
            column.iter().map(|a| a * a / x.len() as f64).sum(),
        );
    }
    for ((&fitted, &r), &y) in fit.iter().zip(&state.residual).zip(y) {
        close(y - fitted, r);
    }
}

// Enumerate knots and affine intervals directly instead of following the
// production kernel's cumulative-threshold scan.
fn independent_scalar(q: f64, b: f64, current: f64, knots: &[(f64, f64)]) -> f64 {
    let mut levels: Vec<_> = knots.iter().map(|&(t, _)| t).collect();
    levels.sort_by(f64::total_cmp);
    levels.dedup();
    let mut candidates = levels.clone();
    candidates.push(current);
    if q > 0.0 {
        for i in 0..=levels.len() {
            let left = if i == 0 {
                f64::NEG_INFINITY
            } else {
                levels[i - 1]
            };
            let right = levels.as_slice().get(i).copied().unwrap_or(f64::INFINITY);
            let slope: f64 = knots
                .iter()
                .map(|&(t, w)| if t <= left { w } else { -w })
                .sum();
            candidates.push(((b - slope) / q).clamp(left, right));
        }
    }
    let value = |z: f64| {
        0.5 * q * z * z - b * z + knots.iter().map(|&(t, w)| w * (z - t).abs()).sum::<f64>()
    };
    candidates
        .into_iter()
        .min_by(|&a, &z| {
            value(a)
                .total_cmp(&value(z))
                .then((a - current).abs().total_cmp(&(z - current).abs()))
        })
        .unwrap()
}

#[test]
fn scalar_kinks_crossings_and_median_intervals() {
    let cases = [
        (
            1.0,
            -8.0,
            0.0,
            vec![(-1.5, 0.25), (0.0, 0.5), (0.75, 0.5), (2.0, 0.75)],
        ),
        (
            1.0,
            8.0,
            0.0,
            vec![(-1.5, 0.25), (0.0, 0.5), (0.75, 0.5), (2.0, 0.75)],
        ),
        (1.0, 0.0, 2.0, vec![(0.0, 0.2), (0.0, 0.3)]),
        (0.0, 0.0, 0.5, vec![(-1.0, 0.1), (1.0, 0.1)]),
        (0.0, 0.0, 4.0, vec![(-1.0, 0.1), (1.0, 0.1)]),
        (0.0, 0.0, -4.0, vec![(-1.0, 0.1), (1.0, 0.1)]),
        (0.0, 0.0, 4.0, vec![(0.0, 0.3)]),
        (0.0, 0.0, 4.0, vec![]),
        (1e-12, 1e-12, 0.0, vec![]),
    ];
    for (q, b, current, mut knots) in cases {
        let expected = independent_scalar(q, b, current, &knots);
        let actual = cluster::scalar_minimum(q, b, current, &mut knots, 0).unwrap();
        close(actual, expected);
    }
    assert!(cluster::scalar_minimum(0.0, 1.0, 0.0, &mut vec![], 0).is_err());
    // Compensation preserves the whole median interval despite unequal summands.
    close(
        cluster::scalar_minimum(
            0.0,
            0.0,
            0.5,
            &mut vec![(0.0, 0.1), (0.0, 0.2), (1.0, 0.3)],
            0,
        )
        .unwrap(),
        0.5,
    );
}

#[test]
fn canonical_signed_membership_and_canceling_aggregates() {
    let rows = vec![
        vec![1.0, -1.0, 2.0, -2.0, 3.0, 4.0],
        vec![2.0, -2.0, 1.0, -1.0, 1.0, 2.0],
    ];
    let x = Matrix::from_rows(&rows.iter().map(Vec::as_slice).collect::<Vec<_>>());
    let matrix = LazyMatrix::from_parts(&x, None, None);
    let beta = [
        0.0,
        -0.0,
        -1.0,
        -1.0,
        1.0,
        f64::from_bits(1.0_f64.to_bits() + 1),
    ];
    let y = [1.0, -1.0];
    let r = residual(&rows, &y, &beta);
    let mut state = State::new(2, 6);
    state.rebuild(&matrix, &beta, &r, 0).unwrap();
    assert_eq!(state.blocks.len(), 4);
    assert_eq!(state.blocks[0].curvature, 0.0);
    assert_eq!(state.blocks[1].curvature, 0.0);
    assert_state(&state, &rows, &y, &beta);
    state
        .sweep::<std::convert::Infallible>(
            &mut beta.clone(),
            &mut 0.0,
            false,
            &ClusteredLasso::new(0.1, 0.2),
            1,
        )
        .unwrap();
}

#[test]
fn sweeps_match_rebuilding_each_scalar_restriction() {
    for seed in 0..12 {
        let rows: Vec<Vec<f64>> = (0..6)
            .map(|i| {
                (0..7)
                    .map(|j| ((i * 13 + j * 7 + seed * 11) % 31) as f64 / 7.0 - 2.0)
                    .collect()
            })
            .collect();
        let y: Vec<_> = (0..6)
            .map(|i| ((i * 7 + seed * 3) % 13) as f64 - 6.0)
            .collect();
        let x = Matrix::from_rows(&rows.iter().map(Vec::as_slice).collect::<Vec<_>>());
        let matrix = LazyMatrix::from_parts(&x, None, None);
        for (l1, fusion) in [(0.1, 0.05), (0.0, 0.05), (0.1, 0.0), (0.0, 0.0)] {
            let mut beta = vec![1.0, 1.0, 0.0, 0.0, -2.0, -2.0, 3.0];
            let mut state = State::new(6, 7);
            state
                .rebuild(&matrix, &beta, &residual(&rows, &y, &beta), 0)
                .unwrap();
            for sweep in 1..=5 {
                let mut reference = beta.clone();
                let representatives: Vec<_> = state.blocks.iter().map(|b| b.members[0]).collect();
                let mut visited = [false; 7];
                for representative in representatives {
                    if visited[representative] {
                        continue;
                    }
                    let current = reference[representative];
                    let members: Vec<_> = (0..7).filter(|&j| reference[j] == current).collect();
                    let column: Vec<_> = rows
                        .iter()
                        .map(|row| members.iter().map(|&j| row[j]).sum::<f64>())
                        .collect();
                    let r = residual(&rows, &y, &reference);
                    let q = column.iter().map(|a| a * a / 6.0).sum();
                    let b = column
                        .iter()
                        .zip(&r)
                        .map(|(&a, &r)| a * (r + a * current) / 6.0)
                        .sum();
                    let mut knots = vec![(0.0, l1 * members.len() as f64)];
                    knots.extend(
                        (0..7)
                            .filter(|j| !members.contains(j))
                            .map(|j| (reference[j], fusion * members.len() as f64)),
                    );
                    let z = independent_scalar(q, b, current, &knots);
                    for &j in &members {
                        reference[j] = z;
                    }
                    for j in 0..7 {
                        visited[j] |= reference[j] == z;
                    }
                }
                let before = objective(&rows, &y, &beta, l1, fusion);
                state
                    .sweep::<std::convert::Infallible>(
                        &mut beta,
                        &mut 0.0,
                        false,
                        &ClusteredLasso::new(l1, fusion),
                        sweep,
                    )
                    .unwrap();
                for (&actual, &expected) in beta.iter().zip(&reference) {
                    close(actual, expected);
                }
                assert!(objective(&rows, &y, &beta, l1, fusion) <= before + 1e-11);
                assert_state(&state, &rows, &y, &beta);
            }
        }
    }
}

#[test]
fn pg_splits_nonzero_blocks_and_stages_cache_rebuilds() {
    let x = Matrix::from_rows(&[&[1.0, 0.0], &[0.0, 2.0]]);
    let problem = Problem::new(
        &x,
        Gaussian::new(&[-2.0, 4.0]).fit_intercept(false),
        ClusteredLasso::new(0.05, 0.1),
    )
    .normalize(crate::Normalization::None);
    let mut design = NativeDesign::<_, Vec<f64>>::new(problem.prepare(false).unwrap());
    let mut work = Workspace::new(2, 2);
    work.coefficients.fill(1.0);
    let options = Hybrid::<Vec<f64>>::new()
        .max_iterations(1)
        .check_every(1)
        .step_policy(HybridStep::FixedBound);
    let solution = iterate(
        &mut design,
        *problem.datafit(),
        problem.penalty(),
        &options,
        1e-12,
        &mut work,
    )
    .unwrap();
    assert_ne!(solution.labels[0], solution.labels[1]);
    assert_eq!(solution.pg_steps, 1);
    assert_eq!(solution.coordinate_sweeps, 0);
    let saved = work.state.labels.clone();
    work.candidate.fill(f64::MAX);
    // A failed staged rebuild must leave the committed state untouched.
    let bad = Matrix::from_rows(&[&[f64::MAX, f64::MAX], &[f64::MAX, f64::MAX]]);
    assert!(
        work.pending
            .rebuild(
                &LazyMatrix::from_parts(&bad, None, None),
                &work.candidate,
                &[0.0, 0.0],
                1
            )
            .is_err()
    );
    assert_eq!(work.state.labels, saved);
    let _ = Matrix::empty(1, 0);
}

#[test]
fn unchanged_cluster_sweeps_wait_for_the_scheduled_pg_split() {
    let x = Matrix::from_rows(&[&[1.0, 0.0], &[0.0, 1.0]]);
    let problem = Problem::new(
        &x,
        Gaussian::new(&[-2.0, 2.0]).fit_intercept(false),
        ClusteredLasso::new(0.05, 0.1),
    )
    .normalize(crate::Normalization::None);
    let mut before = None;
    for budget in [6, 7] {
        let mut design = NativeDesign::<_, Vec<f64>>::new(problem.prepare(false).unwrap());
        let mut work = Workspace::new(2, 2);
        work.coefficients.copy_from_slice(&[2.0 / 3.0, -2.0 / 3.0]);
        let options = Hybrid::<Vec<f64>>::new()
            .max_iterations(budget)
            .check_every(1)
            .tolerance(1e-14)
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
        if budget == 6 {
            assert_eq!(solution.labels[0], solution.labels[1]);
            assert_eq!(solution.proximal.termination, Termination::IterationLimit);
            before = Some(solution.proximal.coefficients);
        } else {
            assert_ne!(solution.labels[0], solution.labels[1]);
            assert_ne!(Some(solution.proximal.coefficients), before);
        }
    }
}

#[test]
fn a_moving_zero_block_merges_and_recomputes_curvature_cross_terms() {
    let x = Matrix::from_rows(&[&[1.0, 2.0]]);
    let matrix = LazyMatrix::from_parts(&x, None, None);
    let mut beta = [0.0, 1.0];
    let mut state = State::new(1, 2);
    state.rebuild(&matrix, &beta, &[1.0], 0).unwrap();
    assert_eq!(state.blocks.iter().map(|b| b.curvature).sum::<f64>(), 5.0);
    state
        .sweep::<std::convert::Infallible>(
            &mut beta,
            &mut 0.0,
            false,
            &ClusteredLasso::new(0.0, 10.0),
            1,
        )
        .unwrap();
    assert_eq!(beta, [1.0, 1.0]);
    assert_eq!(state.blocks.len(), 1);
    assert_eq!(state.blocks[0].column, [3.0]);
    assert_eq!(state.blocks[0].curvature, 9.0);
    assert_state(&state, &[vec![1.0, 2.0]], &[3.0], &beta);
}

#[test]
fn rebuilds_reuse_columns_and_invalidate_membership_even_at_equal_cluster_counts() {
    let x = Matrix::from_rows(&[&[1.0, 2.0, 3.0], &[2.0, -1.0, 0.0]]);
    let matrix = LazyMatrix::from_parts(&x, None, None);
    let mut state = State::new(2, 3);
    state
        .rebuild(&matrix, &[1.0, 1.0, 2.0], &[0.0; 2], 0)
        .unwrap();
    let mut addresses: Vec<_> = state.blocks.iter().map(|b| b.column.as_ptr()).collect();
    addresses.sort();
    let fixed = (state.residual.as_ptr(), state.labels.as_ptr());
    state
        .rebuild(&matrix, &[1.0, 2.0, 2.0], &[1.0; 2], 1)
        .unwrap();
    assert_eq!(state.blocks.len(), 2);
    assert_eq!(state.blocks[0].members, [0]);
    assert_eq!(state.blocks[1].members, [1, 2]);
    assert_eq!(state.blocks[0].column, [1.0, 2.0]);
    assert_eq!(state.blocks[1].column, [5.0, -1.0]);
    let mut rebuilt: Vec<_> = state.blocks.iter().map(|b| b.column.as_ptr()).collect();
    rebuilt.sort();
    assert_eq!(addresses, rebuilt);
    assert_eq!(fixed, (state.residual.as_ptr(), state.labels.as_ptr()));
    assert_eq!(state.residual, [1.0; 2]);
}

#[test]
fn failed_products_preserve_backend_errors_and_discard_partial_outputs() {
    struct Broken {
        transpose: bool,
    }
    impl ProximalDesign for Broken {
        type Error = std::io::Error;
        fn nrows(&self) -> usize {
            1
        }
        fn ncols(&self) -> usize {
            1
        }
        fn forward(&mut self, _: &[f64], output: &mut [f64]) -> Result<(), FitError<Self::Error>> {
            if self.transpose {
                output[0] = 0.0;
                return Ok(());
            }
            output[0] = f64::NAN;
            Err(FitError::Backend {
                operation: "injected hybrid forward",
                source: std::io::Error::other("partial forward"),
            })
        }
        fn transpose(
            &mut self,
            _: &[f64],
            output: &mut [f64],
        ) -> Result<(), FitError<Self::Error>> {
            output[0] = f64::NAN;
            Err(FitError::Backend {
                operation: "injected hybrid transpose",
                source: std::io::Error::other("partial transpose"),
            })
        }
        fn into_preprocessing(self: Box<Self>) -> crate::Preprocessing {
            unreachable!()
        }
    }
    for transpose in [false, true] {
        let error = Workspace::new(1, 1)
            .prepare(&mut Broken { transpose }, Gaussian::new(&[1.0]), 0)
            .unwrap_err();
        let FitError::Backend { operation, source } = error else {
            panic!("partial output must not replace the backend error")
        };
        assert_eq!(
            operation,
            if transpose {
                "injected hybrid transpose"
            } else {
                "injected hybrid forward"
            }
        );
        assert_eq!(
            source.to_string(),
            if transpose {
                "partial transpose"
            } else {
                "partial forward"
            }
        );
    }
}
