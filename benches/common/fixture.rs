use faer::Mat;
use faer::sparse::{SparseColMat, Triplet};

pub fn dataset(n: usize, p: usize) -> (Mat<f64>, SparseColMat<usize, f64>, Vec<f64>) {
    let dense = Mat::from_fn(n, p, |i, j| {
        let index = (i * 17 + j * 31) % 97;
        if index < 8 { index as f64 - 3.0 } else { 0.0 }
    });
    let mut triplets = Vec::new();
    for j in 0..p {
        for i in 0..n {
            if dense[(i, j)] != 0.0 {
                triplets.push(Triplet::new(i, j, dense[(i, j)]));
            }
        }
    }
    let sparse = SparseColMat::<usize, f64>::try_new_from_triplets(n, p, &triplets).unwrap();
    let y: Vec<_> = (0..n)
        .map(|i| {
            1.0 + 2.0 * dense[(i, 0)] - 1.5 * dense[(i, 2)]
                + 0.75 * dense[(i, 7)]
                + 0.01 * ((i * 7) % 11) as f64
        })
        .collect();
    (dense, sparse, y)
}
