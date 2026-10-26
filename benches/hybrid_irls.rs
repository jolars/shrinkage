//! Complete logistic fits, including preparation, reweighting, and diagnostics.

use criterion::{Criterion, criterion_group, criterion_main};
use faer::Col;
use shrinkage::{
    Binomial, ChainFusedLasso, ClusteredLasso, Hybrid, Normalization, Problem, ProximalGradient,
    Slope, Termination,
};
use std::{hint::black_box, time::Duration};

#[path = "common/hybrid_irls.rs"]
mod fixture;

fn solves(c: &mut Criterion) {
    for (n, p) in [(256, 32), (512, 64)] {
        let (x, y) = fixture::dataset(n, p);
        for (regime, fusion) in [("weak", 0.0), ("fused", 0.05)] {
            macro_rules! benchmark {
                ($name:literal, $penalty:expr) => {{
                    let penalty = $penalty;
                    let problem = Problem::new(&x, Binomial::new(&y), penalty.clone()).normalize(Normalization::None);
                    let solver = Hybrid::<Col<f64>>::new().tolerance(1e-7).max_iterations(100_000);
                    let pg = ProximalGradient::<Col<f64>>::new().tolerance(1e-7).max_iterations(100_000);
                    let reference = problem.fit_with(&pg).unwrap();
                    let fit = problem.fit_with(&solver).unwrap();
                    assert_eq!(reference.termination(), Termination::Converged);
                    assert_eq!(fit.termination(), Termination::Converged);
                    assert!((fit.objective()-reference.objective()).abs() < 1e-8);
                    fixture::check(&x,&y,&fit,&penalty);
                    eprintln!("{},{regime},{n},{p},clusters={},outer={},pg={},sweeps={},checks={},inner_backtracks={},outer_backtracks={}", $name, fit.cluster_values().len(), fit.outer_iterations(), fit.pg_steps(), fit.coordinate_sweeps(), fit.diagnostic_checks(), fit.backtracks(), fit.outer_backtracks());
                    let mut group = c.benchmark_group(format!("logistic_{}_{regime}_{n}x{p}", $name));
                    group.sample_size(10);
                    group.warm_up_time(Duration::from_millis(500));
                    group.measurement_time(Duration::from_secs(1));
                    group.bench_function("hybrid_irls", |b| b.iter(|| black_box(&problem).fit_with(black_box(&solver)).unwrap()));
                    group.bench_function("pg", |b| b.iter(|| black_box(&problem).fit_with(black_box(&pg)).unwrap()));
                    for (label, budget) in [("setup_first_pg", 1), ("first_rebuilds", 7)] {
                        let limited = solver.clone().max_iterations(budget);
                        group.bench_function(label, |b| b.iter(|| black_box(&problem).fit_with(black_box(&limited)).unwrap()));
                    }
                    group.finish();
                }};
            }
            benchmark!("clustered", ClusteredLasso::new(0.001, fusion / p as f64));
            benchmark!("chain", ChainFusedLasso::new(0.001, fusion));
            benchmark!(
                "slope",
                Slope::new(
                    (0..p)
                        .map(|j| 0.001 + fusion * (p - j) as f64 / p as f64)
                        .collect()
                )
            );
        }
    }
}

criterion_group!(benches, solves);
criterion_main!(benches);
