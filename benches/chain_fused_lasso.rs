//! Proximal operations and complete Gaussian solves, including rebuilds.

use std::{hint::black_box, time::Duration};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use faer::Col;
use shrinkage::{
    ChainFusedLasso, Gaussian, Hybrid, HybridStep, Normalization, Penalty, Problem,
    ProximalGradient, ProximalPenalty, Termination,
};

#[path = "common/chain.rs"]
mod fixture;

fn operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("chain_fused_lasso_penalty");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(1));
    for p in [64, 1_024, 16_384] {
        for (name, values) in [
            (
                "oscillating",
                (0..p)
                    .map(|j| ((j * 37 % 103) as f64 - 51.0) / 10.0)
                    .collect::<Vec<_>>(),
            ),
            ("ramp", (0..p).map(|j| j as f64 / p as f64).collect()),
        ] {
            let weights = (0..p - 1)
                .map(|j| {
                    if j % 31 == 0 {
                        0.0
                    } else {
                        0.25 + (j % 7) as f64 / 4.0
                    }
                })
                .collect();
            let penalty = ChainFusedLasso::new(0.1, 0.5).with_weights(weights);
            let mut output = vec![0.0; p];
            group.bench_function(BenchmarkId::new(format!("prox_{name}"), p), |b| {
                b.iter(|| {
                    penalty
                        .prox(black_box(&values), 0.5, black_box(&mut output))
                        .unwrap()
                })
            });
            group.bench_function(BenchmarkId::new(format!("value_{name}"), p), |b| {
                b.iter(|| penalty.value(black_box(&values)).unwrap())
            });
        }
    }
    group.finish();
}

fn solves(c: &mut Criterion) {
    for (n, p) in [(256, 64), (512, 128)] {
        let (x, y) = fixture::dataset(n, p);
        for (name, fusion) in [("segmented", 0.05), ("weak_fusion", 1e-6)] {
            let penalty = ChainFusedLasso::new(0.01, fusion);
            let problem = Problem::new(&x, Gaussian::new(&y).fit_intercept(false), penalty.clone())
                .normalize(Normalization::None);
            let pg = ProximalGradient::<Col<f64>>::new()
                .tolerance(1e-8)
                .max_iterations(100_000);
            let reference = problem.fit_with(&pg).unwrap();
            assert_eq!(reference.termination(), Termination::Converged);
            assert!(fixture::mapping(&x, &y, reference.coefficients(), &penalty) <= 5e-7);
            let mut group = c.benchmark_group(format!("chain_fused_lasso_{name}_{n}x{p}"));
            group.sample_size(10);
            group.warm_up_time(Duration::from_millis(500));
            group.measurement_time(Duration::from_secs(1));
            group.bench_function("pg", |b| {
                b.iter(|| black_box(&problem).fit_with(black_box(&pg)).unwrap())
            });
            for (label, policy) in [
                ("hybrid_backtracking", HybridStep::default()),
                ("hybrid_fixed", HybridStep::FixedBound),
            ] {
                let solver = Hybrid::<Col<f64>>::new()
                    .tolerance(1e-8)
                    .max_iterations(100_000)
                    .step_policy(policy);
                let fit = problem.fit_with(&solver).unwrap();
                assert_eq!(fit.termination(), Termination::Converged);
                assert!(
                    (fit.objective() - reference.objective()).abs()
                        <= 1e-8 * (1.0 + reference.objective().abs())
                );
                assert!(fixture::mapping(&x, &y, fit.coefficients(), &penalty) <= 5e-7);
                eprintln!(
                    "{name},{n},{p},{label},segments={},pg_steps={},sweeps={},checks={},backtracks={}",
                    fit.cluster_values().len(),
                    fit.pg_steps(),
                    fit.coordinate_sweeps(),
                    fit.diagnostic_checks(),
                    fit.backtracks()
                );
                group.bench_function(label, |b| {
                    b.iter(|| black_box(&problem).fit_with(black_box(&solver)).unwrap())
                });
            }
            // These bounded runs include preparation and final diagnostics. The
            // second also includes a subsequent PG rebuild after five sweeps.
            for (label, budget) in [("setup_and_first_pg", 1), ("setup_sweeps_and_second_pg", 7)] {
                let solver = Hybrid::<Col<f64>>::new().max_iterations(budget);
                group.bench_function(label, |b| {
                    b.iter(|| black_box(&problem).fit_with(black_box(&solver)).unwrap())
                });
            }
            group.finish();
        }
    }
}

criterion_group!(benches, operations, solves);
criterion_main!(benches);
