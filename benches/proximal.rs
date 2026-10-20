//! End-to-end proximal fitting through dense and CSC native products.

use std::{hint::black_box, time::Duration};

use criterion::{Criterion, criterion_group, criterion_main};
use faer::Col;
use shrinkage::{
    AxisPenalty, CoefficientLayout, DisjointGroups, ElasticNet, Gaussian, GroupLasso, L1,
    MatrixDesign, MultivariateGaussian, PenaltyAxis, Problem, ProximalGradient, ProximalPenalty,
    Ridge, RuntimeProblem, Termination, UncoveredPolicy,
};

#[path = "common/fixture.rs"]
mod fixture;

fn benchmark_penalty<P: ProximalPenalty + Clone + 'static>(
    c: &mut Criterion,
    name: &str,
    penalty: P,
) {
    let (dense, sparse, y) = fixture::dataset(1_024, 64);
    let solver = ProximalGradient::<Col<f64>>::new();
    let dense_problem = Problem::new(&dense, Gaussian::new(&y), penalty.clone());
    let sparse_problem = Problem::new(&sparse, Gaussian::new(&y), penalty.clone());
    let runtime_solver = ProximalGradient::new();
    let runtime_dense = RuntimeProblem::new(
        Box::new(MatrixDesign::<_, Col<f64>>::new(&dense)),
        Box::new(Gaussian::new(&y)),
        Box::new(penalty.clone()),
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
    let groups = DisjointGroups::new(
        64,
        (0..8).map(|g| (8 * g..8 * (g + 1)).collect()).collect(),
        vec![1.0; 8],
        UncoveredPolicy::Reject,
    )
    .unwrap();
    benchmark_penalty(c, "group_lasso", GroupLasso::new(0.03, groups));
}

fn benchmark_multivariate(c: &mut Criterion) {
    let (dense, sparse, y) = fixture::dataset(1_024, 64);
    let responses: Vec<_> = (0..4)
        .flat_map(|response| {
            y.iter()
                .map(move |value| value * (response + 1) as f64 + response as f64)
        })
        .collect();
    let layout = CoefficientLayout::new(64, 4).unwrap();
    let datafit = MultivariateGaussian::new(&responses, 1_024, 4);
    let solver = ProximalGradient::<Col<f64>>::new();
    for (axis, name) in [
        (PenaltyAxis::Features, "features"),
        (PenaltyAxis::Responses, "responses"),
    ] {
        let dense_problem = Problem::new(
            &dense,
            datafit,
            AxisPenalty::new(layout, axis, ElasticNet::new(0.03, 0.1)),
        );
        let sparse_problem = Problem::new(
            &sparse,
            datafit,
            AxisPenalty::new(layout, axis, ElasticNet::new(0.03, 0.1)),
        );
        assert_eq!(
            dense_problem.fit_with(&solver).unwrap().termination(),
            Termination::Converged
        );
        assert_eq!(
            sparse_problem.fit_with(&solver).unwrap().termination(),
            Termination::Converged
        );
        let mut group = c.benchmark_group(format!("multivariate_{name}_1024x64x4"));
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
        group.finish();
    }
}

criterion_group!(benches, benchmark_proximal, benchmark_multivariate);
criterion_main!(benches);
