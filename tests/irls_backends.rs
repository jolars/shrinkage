//! Coordinate IRLS through versioned dense and sparse integrations.

#![cfg(any(
    feature = "faer_v0_24",
    feature = "nalgebra_v0_34",
    feature = "ndarray_v0_17",
    feature = "sprs_v0_11"
))]

use shrinkage::lazymatrix::{ColumnStats, RawColumns};
use shrinkage::{
    Binomial, Centering, ElasticNet, Irls, Normalization, PathOptions, Problem, Scaling,
    Termination,
};

const ROWS: [[f64; 4]; 6] = [
    [0.0, 1.0, 5.0, 0.0],
    [1.0, 0.0, 5.0, 0.0],
    [2.0, 3.0, 5.0, 0.0],
    [0.0, -1.0, 5.0, 0.0],
    [4.0, 0.0, 5.0, 0.0],
    [3.0, 2.0, 5.0, 0.0],
];
const Y: [f64; 6] = [0.0, 0.2, 1.0, 0.0, 0.9, 0.7];

fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-7 * (1.0 + b.abs()), "{a} != {b}");
}

fn compare<A, B>(a: &A, b: &B)
where
    A: RawColumns<f64> + ColumnStats<f64>,
    B: RawColumns<f64> + ColumnStats<f64>,
    A::Error: std::fmt::Debug,
    B::Error: std::fmt::Debug,
{
    let solver = Irls::new().tolerance(1e-10);
    for intercept in [false, true] {
        for normalization in [
            Normalization::None,
            Normalization::Auto,
            Normalization::Center,
            Normalization::Standardize,
            Normalization::MinMax,
            Normalization::MaxAbs,
            Normalization::L1,
            Normalization::L2,
            Normalization::Custom {
                center: Centering::Min,
                scale: Scaling::L2,
            },
        ] {
            let datafit = Binomial::new(&Y).fit_intercept(intercept);
            let penalty = ElasticNet::new(0.03, 0.1);
            let first = Problem::new(a, datafit, penalty)
                .normalize(normalization)
                .fit_with(&solver)
                .unwrap();
            let second = Problem::new(b, datafit, penalty)
                .normalize(normalization)
                .fit_with(&solver)
                .unwrap();
            for fit in [&first, &second] {
                assert_eq!(fit.termination(), Termination::Converged);
                assert!(fit.kkt_violation() <= 1e-10);
            }
            close(first.objective(), second.objective());
            close(first.intercept(), second.intercept());
            for (&v, &w) in first.coefficients().iter().zip(second.coefficients()) {
                close(v, w);
            }
            for (v, w) in first
                .predict(a)
                .unwrap()
                .into_iter()
                .zip(second.predict(b).unwrap())
            {
                close(v, w);
            }
        }
        let datafit = Binomial::new(&Y).fit_intercept(intercept);
        let penalty = ElasticNet::from_alpha(0.0, 0.5);
        let options = PathOptions::new().nlambda(4).lambda_min_ratio(0.05);
        let first = Problem::new(a, datafit, penalty)
            .fit_regularization_path_with(&solver, &options)
            .unwrap();
        let second = Problem::new(b, datafit, penalty)
            .fit_regularization_path_with(&solver, &options)
            .unwrap();
        for (a, b) in first.fits().iter().zip(second.fits()) {
            assert_eq!(a.termination(), Termination::Converged);
            assert_eq!(b.termination(), Termination::Converged);
            close(a.objective(), b.objective());
        }
        for (&a, &b) in first
            .lambdas()
            .unwrap()
            .iter()
            .zip(second.lambdas().unwrap())
        {
            close(a, b);
        }
    }
}

#[cfg(feature = "faer_v0_24")]
#[test]
fn faer_dense_csc_and_transposed_views_agree() {
    let dense = faer::Mat::from_fn(6, 4, |i, j| ROWS[i][j]);
    let triplets: Vec<_> = ROWS
        .iter()
        .enumerate()
        .flat_map(|(i, row)| {
            row.iter().enumerate().filter_map(move |(j, &v)| {
                (v != 0.0 || i == 0).then_some(faer::sparse::Triplet::new(i, j, v))
            })
        })
        .collect();
    let sparse =
        faer::sparse::SparseColMat::<usize, f64>::try_new_from_triplets(6, 4, &triplets).unwrap();
    compare(&dense, &sparse);
    let transposed = faer::Mat::from_fn(4, 6, |i, j| ROWS[j][i]);
    compare(&transposed.transpose(), &sparse);
}

#[cfg(feature = "nalgebra_v0_34")]
#[test]
fn nalgebra_dense_and_csc_agree() {
    let dense = nalgebra::DMatrix::from_fn(6, 4, |i, j| ROWS[i][j]);
    let mut coo = nalgebra_sparse::CooMatrix::new(6, 4);
    for (i, row) in ROWS.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            if v != 0.0 || i == 0 {
                coo.push(i, j, v);
            }
        }
    }
    compare(&dense, &nalgebra_sparse::CscMatrix::from(&coo));
}

#[cfg(feature = "ndarray_v0_17")]
#[test]
fn ndarray_layouts_and_strided_views_agree() {
    use ndarray::{Array2, Axis, ShapeBuilder, Slice};
    let dense = Array2::from_shape_fn((6, 4), |(i, j)| ROWS[i][j]);
    let column_major = Array2::from_shape_fn((6, 4).f(), |(i, j)| ROWS[i][j]);
    compare(&dense, &column_major);
    let padded = Array2::from_shape_fn((12, 8), |(i, j)| {
        if i % 2 == 0 && j % 2 == 0 {
            ROWS[i / 2][j / 2]
        } else {
            f64::NAN
        }
    });
    let mut view = padded.view();
    view.slice_axis_inplace(Axis(0), Slice::new(0, None, 2));
    view.slice_axis_inplace(Axis(1), Slice::new(0, None, 2));
    compare(&dense, &view);
}

#[cfg(feature = "sprs_v0_11")]
#[test]
fn checked_sprs_owned_and_borrowed_csc_agree() {
    use shrinkage::lazymatrix::SprsCsc;
    let mut triplet = sprs::TriMat::new((6, 4));
    for (i, row) in ROWS.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            if v != 0.0 || i == 0 {
                triplet.add_triplet(i, j, v);
            }
        }
    }
    let owned = SprsCsc::try_new(triplet.to_csc::<usize>()).unwrap();
    let borrowed = SprsCsc::try_new(owned.as_inner().view()).unwrap();
    compare(&owned, &borrowed);
}
