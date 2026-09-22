//! Verify prediction-preserving normalization with dense and sparse CSC input.

use faer::sparse::{SparseColMat, Triplet};
use faer::{Col, Mat};
use shrinkage::lazymatrix::{Centering, LazyMatrix, MatVec, Normalization, Scaling};

fn main() {
    let raw = [[1.0, 0.0], [0.0, 3.0], [4.0, 2.0]];
    let dense = Mat::from_fn(3, 2, |i, j| raw[i][j]);
    let sparse = SparseColMat::<usize, f64>::try_new_from_triplets(
        3,
        2,
        &[
            Triplet::new(0, 0, 1.0),
            Triplet::new(2, 0, 4.0),
            Triplet::new(1, 1, 3.0),
            Triplet::new(2, 1, 2.0),
        ],
    )
    .expect("valid CSC triplets");

    let spec = Normalization::new(Centering::Mean, Scaling::Sd);
    let dense = LazyMatrix::new(&dense, spec).expect("in-memory statistics are infallible");
    let sparse = LazyMatrix::new(&sparse, spec).expect("in-memory statistics are infallible");
    let theta = Col::from_fn(2, |j| [0.75, -1.25][j]);
    let normalized_intercept = 0.5;
    let dense_prediction = dense
        .matvec(&theta)
        .expect("in-memory product is infallible");
    let sparse_prediction = sparse
        .matvec(&theta)
        .expect("in-memory product is infallible");

    let centers = dense.centers().expect("mean centering is enabled");
    let scales = dense
        .scales()
        .expect("standard-deviation scaling is enabled");
    let beta = [theta[0] / scales[0], theta[1] / scales[1]];
    let intercept = normalized_intercept - beta[0] * centers[0] - beta[1] * centers[1];

    for i in 0..raw.len() {
        let original_prediction = raw[i][0] * beta[0] + raw[i][1] * beta[1] + intercept;
        let normalized_prediction = dense_prediction[i] + normalized_intercept;
        assert!((dense_prediction[i] - sparse_prediction[i]).abs() < 1e-12);
        assert!((original_prediction - normalized_prediction).abs() < 1e-12);
    }

    println!("Dense and sparse predictions agree on normalized and original scales.");
}
