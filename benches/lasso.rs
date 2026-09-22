//! End-to-end fitting time, including validation and fitted preprocessing.

use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use faer::Mat;
use faer::sparse::{SparseColMat, Triplet};
use shrinkage::{Lasso, Termination};

fn benchmark_lasso(c: &mut Criterion) {
    let (n, p) = (1_024, 64);
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
    let model = Lasso::new(0.03);
    for status in [
        model.fit(&dense, &y).unwrap().termination(),
        model.fit(&sparse, &y).unwrap().termination(),
    ] {
        assert_eq!(status, Termination::Converged);
    }
    let mut group = c.benchmark_group("lasso_1024x64");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));
    group.bench_function("dense", |b| {
        b.iter(|| model.fit(black_box(&dense), black_box(&y)).unwrap())
    });
    group.bench_function("csc", |b| {
        b.iter(|| model.fit(black_box(&sparse), black_box(&y)).unwrap())
    });
    group.finish();
}

criterion_group!(benches, benchmark_lasso);
criterion_main!(benches);
