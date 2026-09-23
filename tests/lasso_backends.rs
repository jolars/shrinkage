//! Dense, strided, and CSC fits through the optional matrix backends.

#![cfg(any(feature = "faer", feature = "nalgebra"))]

use shrinkage::lazymatrix::{ColumnStats, RawColumns};
use shrinkage::{Centering, Lasso, LassoFit, Normalization, Scaling, Termination};

const ROWS: [[f64; 4]; 6] = [
    [0.0, 1.0, 5.0, 0.0],
    [1.0, 0.0, 5.0, 0.0],
    [2.0, 3.0, 5.0, 0.0],
    [0.0, -1.0, 5.0, 0.0],
    [4.0, 0.0, 5.0, 0.0],
    [3.0, 2.0, 5.0, 0.0],
];
const Y: [f64; 6] = [1.0, 4.0, 2.0, 3.0, 8.0, 5.0];

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-7 * (1.0 + expected.abs()),
        "{actual} != {expected}"
    );
}

fn verify_diagnostics(fit: &LassoFit, intercept: bool, lambda: f64) {
    let centers = fit.preprocessing().centers();
    let scales = fit.preprocessing().scales();
    let residual: Vec<f64> = ROWS
        .iter()
        .zip(Y)
        .map(|(row, target)| {
            target
                - fit.intercept()
                - row
                    .iter()
                    .zip(fit.coefficients())
                    .map(|(x, b)| x * b)
                    .sum::<f64>()
        })
        .collect();
    let penalty: f64 = fit
        .coefficients()
        .iter()
        .enumerate()
        .map(|(j, b)| lambda * (b * scales.map_or(1.0, |s| s[j])).abs())
        .sum();
    close(
        fit.objective(),
        residual.iter().map(|r| r * r).sum::<f64>() / 12.0 + penalty,
    );
    let mut violation = if intercept {
        (residual.iter().sum::<f64>() / 6.0).abs()
    } else {
        0.0
    };
    for (j, &coefficient) in fit.coefficients().iter().enumerate() {
        let c = ROWS
            .iter()
            .zip(&residual)
            .map(|(row, r)| {
                (row[j] - centers.map_or(0.0, |c| c[j])) / scales.map_or(1.0, |s| s[j]) * r
            })
            .sum::<f64>()
            / 6.0;
        let v = if coefficient == 0.0 {
            (c.abs() - lambda).max(0.0)
        } else {
            (c - lambda * coefficient.signum()).abs()
        };
        violation = violation.max(v);
    }
    close(fit.kkt_violation(), violation);
    assert!(violation <= 1.1e-9);
}

fn compare<A, B>(dense: &A, sparse: &B)
where
    A: RawColumns<f64> + ColumnStats<f64>,
    B: RawColumns<f64> + ColumnStats<f64>,
{
    for intercept in [false, true] {
        let mut normalizations = vec![
            Normalization::Auto,
            Normalization::None,
            Normalization::Center,
            Normalization::Standardize,
            Normalization::MinMax,
            Normalization::MaxAbs,
            Normalization::L1,
            Normalization::L2,
        ];
        for center in [Centering::None, Centering::Mean, Centering::Min] {
            for scale in [
                Scaling::None,
                Scaling::Sd,
                Scaling::Range,
                Scaling::MaxAbs,
                Scaling::L1,
                Scaling::L2,
            ] {
                normalizations.push(Normalization::Custom { center, scale });
            }
        }
        for normalization in normalizations {
            let model = Lasso::new(0.15)
                .fit_intercept(intercept)
                .normalize(normalization)
                .tolerance(1e-9)
                .max_iterations(50_000);
            let a = model.fit(dense, &Y).unwrap();
            let b = model.fit(sparse, &Y).unwrap();
            assert_eq!(a.termination(), Termination::Converged);
            assert_eq!(b.termination(), Termination::Converged);
            verify_diagnostics(&a, intercept, 0.15);
            verify_diagnostics(&b, intercept, 0.15);
            for (&a, &b) in a.coefficients().iter().zip(b.coefficients()) {
                close(a, b);
            }
            close(a.intercept(), b.intercept());
            close(a.objective(), b.objective());
            for (a, b) in a
                .predict(sparse)
                .unwrap()
                .iter()
                .zip(b.predict(dense).unwrap())
            {
                close(*a, b);
            }
        }
    }
}

#[cfg(feature = "faer")]
fn faer_matrices() -> (faer::Mat<f64>, faer::sparse::SparseColMat<usize, f64>) {
    let dense = faer::Mat::from_fn(6, 4, |i, j| ROWS[i][j]);
    let mut triplets = Vec::new();
    for (i, row) in ROWS.iter().enumerate() {
        for (j, &value) in row.iter().enumerate() {
            // Include stored zeros to exercise the raw-column contract.
            if value != 0.0 || i == 0 {
                triplets.push(faer::sparse::Triplet::new(i, j, value));
            }
        }
    }
    (
        dense,
        faer::sparse::SparseColMat::try_new_from_triplets(6, 4, &triplets).unwrap(),
    )
}

#[cfg(feature = "nalgebra")]
fn nalgebra_matrices() -> (nalgebra::DMatrix<f64>, nalgebra_sparse::CscMatrix<f64>) {
    let dense = nalgebra::DMatrix::from_fn(6, 4, |i, j| ROWS[i][j]);
    let mut coo = nalgebra_sparse::CooMatrix::new(6, 4);
    for (i, row) in ROWS.iter().enumerate() {
        for (j, &value) in row.iter().enumerate() {
            if value != 0.0 || i == 0 {
                coo.push(i, j, value);
            }
        }
    }
    (dense, nalgebra_sparse::CscMatrix::from(&coo))
}

#[cfg(feature = "faer")]
#[test]
fn faer_dense_csc_and_strided_view_agree() {
    let (dense, sparse) = faer_matrices();
    compare(&dense, &sparse);
    let transposed = faer::Mat::from_fn(4, 6, |i, j| ROWS[j][i]);
    compare(&transposed.transpose(), &sparse);
}

#[cfg(feature = "nalgebra")]
#[test]
fn nalgebra_dense_and_csc_agree() {
    let (dense, sparse) = nalgebra_matrices();
    compare(&dense, &sparse);
}

#[cfg(all(feature = "faer", feature = "nalgebra"))]
#[test]
fn faer_and_nalgebra_agree() {
    let (dense, _) = faer_matrices();
    let (_, sparse) = nalgebra_matrices();
    compare(&dense, &sparse);
}
