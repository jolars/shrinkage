//! Native product buffers, sparse designs, and strided backend views.

#![cfg(any(
    feature = "faer_v0_24",
    feature = "nalgebra_v0_34",
    feature = "ndarray_v0_17",
    feature = "sprs_v0_11"
))]

#[path = "common/matrix.rs"]
mod matrix;

use shrinkage::lazymatrix::{ColumnStats, MatTransposeVecInto, MatVecInto, RawColumns};
use shrinkage::{
    ElasticNet, Gaussian, Normalization, Problem, ProximalGradient, ProximalVector, Termination,
};

const ROWS: [[f64; 3]; 5] = [
    [0.0, 1.0, 5.0],
    [2.0, 0.0, 5.0],
    [4.0, 3.0, 5.0],
    [6.0, 1.0, 5.0],
    [1.0, 0.0, 5.0],
];
const Y: [f64; 5] = [1.0, 2.0, -1.0, 4.0, 2.0];

fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-7 * (1.0 + b.abs()), "{a} != {b}");
}

fn verify<M, V>(design: &M)
where
    M: RawColumns<f64> + ColumnStats<f64> + MatVecInto<V> + MatTransposeVecInto<V>,
    V: ProximalVector,
{
    let rows: Vec<_> = ROWS.iter().map(|row| row.as_slice()).collect();
    let reference = matrix::Matrix::from_rows(&rows);
    for intercept in [false, true] {
        for normalization in [
            Normalization::Auto,
            Normalization::None,
            Normalization::Center,
            Normalization::Standardize,
        ] {
            for (l1, l2) in [(0.1, 0.0), (0.0, 0.3), (0.1, 0.3)] {
                let datafit = Gaussian::new(&Y).fit_intercept(intercept);
                let penalty = ElasticNet::new(l1, l2);
                let actual = Problem::new(design, datafit, penalty)
                    .normalize(normalization)
                    .fit_with(
                        &ProximalGradient::<V>::new()
                            .tolerance(1e-9)
                            .max_iterations(200_000),
                    )
                    .unwrap();
                let expected = Problem::new(&reference, datafit, penalty)
                    .normalize(normalization)
                    .fit_with(
                        &ProximalGradient::<Vec<f64>>::new()
                            .tolerance(1e-9)
                            .max_iterations(200_000),
                    )
                    .unwrap();
                assert_eq!(
                    actual.termination(),
                    Termination::Converged,
                    "intercept={intercept}, normalization={normalization:?}, penalty=({l1}, {l2}), fit={actual:?}"
                );
                assert_eq!(expected.termination(), Termination::Converged);
                close(actual.objective(), expected.objective());
                close(actual.intercept(), expected.intercept());
                for (a, b) in actual.coefficients().iter().zip(expected.coefficients()) {
                    close(*a, *b);
                }
                for (a, b) in actual
                    .predict(design)
                    .unwrap()
                    .iter()
                    .zip(expected.predict(&reference).unwrap())
                {
                    close(*a, b);
                }
                assert!(
                    actual
                        .predict(&matrix::Matrix::empty(0, 3))
                        .unwrap()
                        .is_empty()
                );
            }
        }
    }
}

#[cfg(feature = "faer_v0_24")]
#[test]
fn faer_dense_view_and_csc_use_native_columns() {
    use faer::{
        Col, Mat,
        sparse::{SparseColMat, Triplet},
    };
    let dense = Mat::from_fn(5, 3, |i, j| ROWS[i][j]);
    verify::<_, Col<f64>>(&dense);
    verify::<_, Col<f64>>(&dense.as_ref());
    let triplets: Vec<_> = (0..3)
        .flat_map(|j| {
            (0..5)
                .filter_map(move |i| (ROWS[i][j] != 0.0).then_some(Triplet::new(i, j, ROWS[i][j])))
        })
        .collect();
    let sparse = SparseColMat::<usize, f64>::try_new_from_triplets(5, 3, &triplets).unwrap();
    verify::<_, Col<f64>>(&sparse);
}

#[cfg(feature = "nalgebra_v0_34")]
#[test]
fn nalgebra_dense_view_and_csc_use_native_vectors() {
    use nalgebra::{DMatrix, DVector};
    use nalgebra_sparse::CscMatrix;
    let dense = DMatrix::from_fn(5, 3, |i, j| ROWS[i][j]);
    verify::<_, DVector<f64>>(&dense);
    verify::<_, DVector<f64>>(&dense.view((0, 0), (5, 3)));
    let offsets = vec![0, 4, 7, 12];
    let indices = vec![1, 2, 3, 4, 0, 2, 3, 0, 1, 2, 3, 4];
    let values = vec![2.0, 4.0, 6.0, 1.0, 1.0, 3.0, 1.0, 5.0, 5.0, 5.0, 5.0, 5.0];
    let sparse = CscMatrix::try_from_csc_data(5, 3, offsets, indices, values).unwrap();
    verify::<_, DVector<f64>>(&sparse);
}

#[cfg(feature = "ndarray_v0_17")]
#[test]
fn ndarray_row_major_column_major_and_strided_views_use_arrays() {
    use ndarray::{Array1, Array2, Axis, ShapeBuilder, Slice};
    let row_major = Array2::from_shape_fn((5, 3), |(i, j)| ROWS[i][j]);
    let column_major = Array2::from_shape_fn((5, 3).f(), |(i, j)| ROWS[i][j]);
    let padded = Array2::from_shape_fn((10, 6), |(i, j)| ROWS[i / 2][j / 2]);
    verify::<_, Array1<f64>>(&row_major);
    verify::<_, Array1<f64>>(&column_major);
    let rows = padded.slice_axis(Axis(0), Slice::new(0, None, 2));
    verify::<_, Array1<f64>>(&rows.slice_axis(Axis(1), Slice::new(0, None, 2)));
}

#[cfg(feature = "sprs_v0_11")]
#[test]
fn sprs_owned_and_borrowed_csc_use_vectors() {
    use shrinkage::lazymatrix::SprsCsc;
    let sparse = sprs::CsMat::new_csc(
        (5, 3),
        vec![0, 4, 7, 12],
        vec![1, 2, 3, 4, 0, 2, 3, 0, 1, 2, 3, 4],
        vec![2.0, 4.0, 6.0, 1.0, 1.0, 3.0, 1.0, 5.0, 5.0, 5.0, 5.0, 5.0],
    );
    verify::<_, Vec<f64>>(&SprsCsc::try_new(sparse.view()).unwrap());
    verify::<_, Vec<f64>>(&SprsCsc::try_new(sparse).unwrap());
}
