//! Dense, strided, and CSC fits through the optional matrix backends.

#![cfg(any(
    feature = "faer",
    feature = "nalgebra",
    feature = "ndarray",
    feature = "sprs"
))]

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

fn compare<A, B>(first: &A, second: &B)
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
            let a = model.fit(first, &Y).unwrap();
            let b = model.fit(second, &Y).unwrap();
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
                .predict(second)
                .unwrap()
                .iter()
                .zip(b.predict(first).unwrap())
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

#[cfg(feature = "ndarray")]
fn ndarray_matrix() -> ndarray::Array2<f64> {
    ndarray::Array2::from_shape_fn((6, 4), |(i, j)| ROWS[i][j])
}

#[cfg(feature = "sprs")]
fn sprs_matrix() -> sprs::CsMat<f64> {
    let mut indptr = vec![0];
    let mut indices = Vec::new();
    let mut values = Vec::new();
    for j in 0..4 {
        for (i, row) in ROWS.iter().enumerate() {
            if row[j] != 0.0 || i == 0 {
                indices.push(i);
                values.push(row[j]);
            }
        }
        indptr.push(values.len());
    }
    sprs::CsMat::new_csc((6, 4), indptr, indices, values)
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

#[cfg(feature = "ndarray")]
#[test]
fn ndarray_layouts_and_views_agree() {
    use ndarray::{Array2, Axis, ShapeBuilder, Slice};

    let row_major = ndarray_matrix();
    let column_major = Array2::from_shape_fn((6, 4).f(), |(i, j)| ROWS[i][j]);
    compare(&row_major, &column_major);
    compare(&row_major.view(), &column_major.view());

    // NaNs in the padding expose accidental reads outside the borrowed view.
    let padded = Array2::from_shape_fn((12, 8), |(i, j)| {
        if i % 2 == 0 && j % 2 == 0 {
            ROWS[i / 2][j / 2]
        } else {
            f64::NAN
        }
    });
    let mut strided = padded.view();
    strided.slice_axis_inplace(Axis(0), Slice::new(0, None, 2));
    strided.slice_axis_inplace(Axis(1), Slice::new(0, None, 2));
    compare(&row_major, &strided);
}

#[cfg(feature = "sprs")]
#[test]
fn sprs_owned_and_borrowed_csc_agree() {
    use shrinkage::lazymatrix::SprsCsc;

    let owned = SprsCsc::try_new(sprs_matrix()).unwrap();
    let borrowed = SprsCsc::try_new(owned.as_inner().view()).unwrap();
    compare(&owned, &borrowed);
}

#[cfg(feature = "sprs")]
#[test]
fn sprs_explicit_csr_conversion_preserves_fit() {
    use shrinkage::lazymatrix::SprsCsc;

    let csc = sprs_matrix();
    let csr = csc.to_csr();
    let converted = SprsCsc::try_new(csr.to_csc()).unwrap();
    compare(&SprsCsc::try_new(csc).unwrap(), &converted);
}

#[cfg(all(feature = "faer", feature = "nalgebra"))]
#[test]
fn faer_and_nalgebra_agree() {
    let (dense, _) = faer_matrices();
    let (_, sparse) = nalgebra_matrices();
    compare(&dense, &sparse);
}

#[cfg(all(feature = "faer", feature = "ndarray"))]
#[test]
fn faer_and_ndarray_agree() {
    let (dense, _) = faer_matrices();
    compare(&dense, &ndarray_matrix());
}

#[cfg(all(feature = "faer", feature = "sprs"))]
#[test]
fn faer_and_sprs_agree() {
    let (dense, _) = faer_matrices();
    let sparse = shrinkage::lazymatrix::SprsCsc::try_new(sprs_matrix()).unwrap();
    compare(&dense, &sparse);
}

#[cfg(all(feature = "ndarray", feature = "sprs"))]
#[test]
fn ndarray_and_sprs_agree() {
    let sparse = shrinkage::lazymatrix::SprsCsc::try_new(sprs_matrix()).unwrap();
    compare(&ndarray_matrix(), &sparse);
}
