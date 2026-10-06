//! End-to-end fitting time, including validation and fitted preprocessing.

use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use shrinkage::{Lasso, Termination};

#[path = "common/fixture.rs"]
mod fixture;

fn benchmark_lasso(c: &mut Criterion) {
    let (dense, sparse, y) = fixture::dataset(1_024, 64);
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
