//! Cross-validation uses the same backend statistics and solver products as fitting.

#![cfg(any(
    feature = "faer_v0_24",
    feature = "nalgebra_v0_34",
    feature = "ndarray_v0_17",
    feature = "sprs_v0_11"
))]

#[path = "common/cv_matrix.rs"]
mod matrix;

use std::convert::Infallible;

use shrinkage::lazymatrix::{MatTransposeVecInto, MatVecInto, MatrixShape};
use shrinkage::{
    Binomial, ChainFusedLasso, ClusteredLasso, CrossValidation, FoldAssignments, Gaussian, Hybrid,
    Lasso, PathOptions, Problem, ProximalVector, RowSubset, Slope,
};

const ROWS: [[f64; 3]; 8] = [
    [-2.0, 0.0, 1.0],
    [-1.0, 1.0, 0.0],
    [0.0, 0.0, 2.0],
    [1.0, -1.0, 0.0],
    [2.0, 0.0, 1.0],
    [3.0, 1.0, 0.0],
    [4.0, 0.0, -1.0],
    [0.0, 2.0, 0.0],
];
const Y: [f64; 8] = [-1.0, 0.0, 1.0, 3.0, 2.0, 4.0, 5.0, 1.0];
const BINARY: [f64; 8] = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0];

fn verify<M, V>(x: &M)
where
    M: RowSubset<Error = Infallible> + MatVecInto<V> + MatTransposeVecInto<V>,
    M::Subset: MatVecInto<V> + MatTransposeVecInto<V>,
    V: ProximalVector,
{
    let reference =
        matrix::Matrix::from_rows(&ROWS.iter().map(|r| r.as_slice()).collect::<Vec<_>>());
    let cv = CrossValidation::new(FoldAssignments::new(vec![0, 1, 2, 0, 1, 2, 0, 1]).unwrap());
    let options = PathOptions::new().nlambda(3).lambda_min_ratio(0.2);
    let native = Lasso::new(1.0)
        .cross_validate_regularization_path(x, &Y, &options, &cv)
        .unwrap();
    let expected = Lasso::new(1.0)
        .cross_validate_regularization_path(&reference, &Y, &options, &cv)
        .unwrap();
    compare(native.mean_scores(), expected.mean_scores());
    for (a, b) in native.folds().iter().zip(expected.folds()) {
        compare(a.path().lambdas().unwrap(), b.path().lambdas().unwrap());
        compare(a.scores(), b.scores());
    }
    assert!(
        native
            .fit()
            .predict(&matrix::Matrix::empty(0, 3))
            .unwrap()
            .is_empty()
    );
    assert!(x.select_rows(&[1, 0]).is_err());
    assert!(x.select_rows(&[1, 1]).is_err());
    assert!(x.select_rows(&[ROWS.len()]).is_err());
    assert_eq!(x.select_rows(&[]).unwrap().nrows(), 0);
    for fit_intercept in [false, true] {
        for normalization in [
            shrinkage::Normalization::None,
            shrinkage::Normalization::Standardize,
            shrinkage::Normalization::MinMax,
            shrinkage::Normalization::MaxAbs,
            shrinkage::Normalization::L1,
            shrinkage::Normalization::L2,
            shrinkage::Normalization::Custom {
                center: shrinkage::Centering::Median,
                scale: shrinkage::Scaling::QuantileRange(shrinkage::QuantileRange::default()),
            },
        ] {
            let model = Lasso::new(1.0)
                .fit_intercept(fit_intercept)
                .normalize(normalization)
                .tolerance(1e-9);
            let result = model.cross_validate_path(x, &Y, &[0.3, 0.1], &cv).unwrap();
            for fold in result.folds() {
                let rows = (0..ROWS.len())
                    .filter(|i| !fold.validation_rows().contains(i))
                    .collect::<Vec<_>>();
                let training = x.select_rows(&rows).unwrap();
                let response = rows.iter().map(|&i| Y[i]).collect::<Vec<_>>();
                for (penalty, fit) in fold.path().iter() {
                    let cold = Lasso::new(penalty.lambda())
                        .fit_intercept(fit_intercept)
                        .normalize(normalization)
                        .tolerance(1e-9)
                        .fit(&training, &response)
                        .unwrap();
                    compare(&fit.predict(x).unwrap(), &cold.predict(x).unwrap());
                }
            }
        }
    }
    let native_solver = Hybrid::<V>::new().tolerance(1e-8);
    let reference_solver = Hybrid::<Vec<f64>>::new().tolerance(1e-8);
    macro_rules! check {
        ($family:ident, $response:expr, $penalty:expr) => {{
            let native = Problem::new(x, $family::new($response), $penalty)
                .cross_validate_regularization_path_with(&native_solver, &options, &cv)
                .unwrap();
            let expected = Problem::new(&reference, $family::new($response), $penalty)
                .cross_validate_regularization_path_with(&reference_solver, &options, &cv)
                .unwrap();
            compare(native.mean_scores(), expected.mean_scores());
            compare(native.fit().coefficients(), expected.fit().coefficients());
        }};
    }
    macro_rules! family {
        ($family:ident, $response:expr) => {
            check!($family, $response, ClusteredLasso::new(1.0, 0.2));
            check!($family, $response, ChainFusedLasso::new(1.0, 0.2));
            check!($family, $response, Slope::new(vec![1.0, 0.6, 0.2]));
        };
    }
    family!(Gaussian, &Y);
    family!(Binomial, &BINARY);
}

fn compare(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    for (&a, &b) in a.iter().zip(b) {
        assert!(
            (a - b).abs() < 1e-5 * (1.0 + a.abs() + b.abs()),
            "{a} != {b}"
        );
    }
}

#[cfg(feature = "faer_v0_24")]
#[test]
fn faer_dense_views_and_csc() {
    let dense = faer::Mat::from_fn(8, 3, |i, j| ROWS[i][j]);
    verify::<_, faer::Col<f64>>(&dense);
    verify::<_, faer::Col<f64>>(&dense.as_ref());
    let triplets = ROWS
        .iter()
        .enumerate()
        .flat_map(|(i, row)| {
            row.iter().enumerate().filter_map(move |(j, &value)| {
                (value != 0.0).then_some(faer::sparse::Triplet::new(i, j, value))
            })
        })
        .collect::<Vec<_>>();
    let csc =
        faer::sparse::SparseColMat::<usize, f64>::try_new_from_triplets(8, 3, &triplets).unwrap();
    verify::<_, faer::Col<f64>>(&csc);
}

#[cfg(feature = "nalgebra_v0_34")]
#[test]
fn nalgebra_dense_views_and_csc() {
    let dense = nalgebra::DMatrix::from_fn(8, 3, |i, j| ROWS[i][j]);
    verify::<_, nalgebra::DVector<f64>>(&dense);
    verify::<_, nalgebra::DVector<f64>>(&dense.view((0, 0), (8, 3)));
    let mut coo = nalgebra_sparse::CooMatrix::new(8, 3);
    for (i, row) in ROWS.iter().enumerate() {
        for (j, &value) in row.iter().enumerate() {
            if value != 0.0 {
                coo.push(i, j, value);
            }
        }
    }
    verify::<_, nalgebra::DVector<f64>>(&nalgebra_sparse::CscMatrix::from(&coo));
}

#[cfg(feature = "ndarray_v0_17")]
#[test]
fn ndarray_dense_and_strided_views() {
    let dense = ndarray::Array2::from_shape_fn((8, 3), |(i, j)| ROWS[i][j]);
    verify::<_, ndarray::Array1<f64>>(&dense);
    verify::<_, ndarray::Array1<f64>>(&dense.view());
    let wide = ndarray::Array2::from_shape_fn((8, 6), |(i, j)| ROWS[i][j / 2]);
    verify::<_, ndarray::Array1<f64>>(
        &wide.slice_axis(ndarray::Axis(1), ndarray::Slice::new(0, None, 2)),
    );
}

#[cfg(feature = "sprs_v0_11")]
#[test]
fn sprs_csc_and_borrowed_view() {
    let mut triplets = sprs::TriMat::new((8, 3));
    for (i, row) in ROWS.iter().enumerate() {
        for (j, &value) in row.iter().enumerate() {
            if value != 0.0 {
                triplets.add_triplet(i, j, value);
            }
        }
    }
    let csc = triplets.to_csc::<usize>();
    verify::<_, Vec<f64>>(&shrinkage::lazymatrix::SprsCsc::try_new(csc.view()).unwrap());
    verify::<_, Vec<f64>>(&shrinkage::lazymatrix::SprsCsc::try_new(csc).unwrap());
}
