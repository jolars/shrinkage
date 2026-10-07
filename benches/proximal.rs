//! End-to-end proximal fitting through dense and CSC native products.

use std::{hint::black_box, time::Duration};

use criterion::{Criterion, criterion_group, criterion_main};
use faer::Col;
use shrinkage::{
    ElasticNet, Gaussian, L1, MatrixDesign, Problem, ProximalGradient, ProximalPenalty, Ridge,
    RuntimeProblem, Termination,
};

#[path = "common/fixture.rs"]
mod fixture;

fn benchmark_penalty<P: ProximalPenalty + Copy + 'static>(
    c: &mut Criterion,
    name: &str,
    penalty: P,
) {
    let (dense, sparse, y) = fixture::dataset(1_024, 64);
    let solver = ProximalGradient::<Col<f64>>::new();
    let dense_problem = Problem::new(&dense, Gaussian::new(&y), penalty);
    let sparse_problem = Problem::new(&sparse, Gaussian::new(&y), penalty);
    let runtime_solver = ProximalGradient::new();
    let runtime_dense = RuntimeProblem::new(
        Box::new(MatrixDesign::<_, Col<f64>>::new(&dense)),
        Box::new(Gaussian::new(&y)),
        Box::new(penalty),
    );
    let runtime_sparse = RuntimeProblem::new(
        Box::new(MatrixDesign::<_, Col<f64>>::new(&sparse)),
        Box::new(Gaussian::new(&y)),
        Box::new(penalty),
    );
    for status in [
        dense_problem.fit_with(&solver).unwrap().termination(),
        sparse_problem.fit_with(&solver).unwrap().termination(),
        runtime_dense
            .fit_with(&runtime_solver)
            .unwrap()
            .termination(),
        runtime_sparse
            .fit_with(&runtime_solver)
            .unwrap()
            .termination(),
    ] {
        assert_eq!(status, Termination::Converged);
    }
    let mut group = c.benchmark_group(format!("proximal_{name}_1024x64"));
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));
    group.bench_function("dense", |b| {
        b.iter(|| {
            black_box(&dense_problem)
                .fit_with(black_box(&solver))
                .unwrap()
        })
    });
    group.bench_function("csc", |b| {
        b.iter(|| {
            black_box(&sparse_problem)
                .fit_with(black_box(&solver))
                .unwrap()
        })
    });
    group.bench_function("dense_runtime", |b| {
        b.iter(|| {
            black_box(&runtime_dense)
                .fit_with(black_box(&runtime_solver))
                .unwrap()
        })
    });
    group.bench_function("csc_runtime", |b| {
        b.iter(|| {
            black_box(&runtime_sparse)
                .fit_with(black_box(&runtime_solver))
                .unwrap()
        })
    });
    group.finish();
}

fn benchmark_proximal(c: &mut Criterion) {
    benchmark_penalty(c, "lasso", L1::new(0.03));
    benchmark_penalty(c, "ridge", Ridge::new(0.1));
    benchmark_penalty(c, "elastic_net", ElasticNet::new(0.03, 0.1));
}

criterion_group!(benches, benchmark_proximal);
criterion_main!(benches);
