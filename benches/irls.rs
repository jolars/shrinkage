//! Binomial fits and paths with dense and sparse designs.

use criterion::{Criterion, criterion_group, criterion_main};
use faer::Col;
use shrinkage::{Binomial, ElasticNet, Irls, PathOptions, Problem, ProximalGradient, Termination};
use std::{hint::black_box, time::Duration};

#[path = "common/binomial.rs"]
mod binomial;
#[path = "common/fixture.rs"]
mod fixture;

fn benchmark(c: &mut Criterion) {
    let (dense, sparse, latent) = fixture::dataset(1_024, 64);
    let y = binomial::responses(&latent);
    let penalty = ElasticNet::from_alpha(0.03, 0.8);
    let dense_problem = Problem::new(&dense, Binomial::new(&y), penalty);
    let sparse_problem = Problem::new(&sparse, Binomial::new(&y), penalty);
    let irls = Irls::new();
    let proximal = ProximalGradient::<Col<f64>>::new().max_iterations(100_000);
    let expected = dense_problem.fit_with(&proximal).unwrap();
    let actual = dense_problem.fit_with(&irls).unwrap();
    assert_eq!(expected.termination(), Termination::Converged);
    assert_eq!(actual.termination(), Termination::Converged);
    assert!((expected.objective() - actual.objective()).abs() < 1e-6);
    let mut group = c.benchmark_group("binomial_1024x64");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));
    group.bench_function("irls_dense", |b| {
        b.iter(|| dense_problem.fit_with(black_box(&irls)).unwrap())
    });
    group.bench_function("irls_csc", |b| {
        b.iter(|| sparse_problem.fit_with(black_box(&irls)).unwrap())
    });
    group.bench_function("proximal_dense", |b| {
        b.iter(|| dense_problem.fit_with(black_box(&proximal)).unwrap())
    });
    group.bench_function("proximal_csc", |b| {
        b.iter(|| sparse_problem.fit_with(black_box(&proximal)).unwrap())
    });
    let options = PathOptions::new().nlambda(10).lambda_min_ratio(0.1);
    group.bench_function("irls_path_dense", |b| {
        b.iter(|| {
            dense_problem
                .fit_regularization_path_with(&irls, black_box(&options))
                .unwrap()
        })
    });
    group.bench_function("irls_path_csc", |b| {
        b.iter(|| {
            sparse_problem
                .fit_regularization_path_with(&irls, black_box(&options))
                .unwrap()
        })
    });
    group.bench_function("irls_path_csc_full_sweeps", |b| {
        b.iter(|| {
            sparse_problem
                .fit_regularization_path_with(&irls.working_sets(false), black_box(&options))
                .unwrap()
        })
    });
    group.finish();
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
