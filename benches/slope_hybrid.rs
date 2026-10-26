//! Complete solves at matched accuracy, including preprocessing and rebuilds.

use std::{hint::black_box, time::Duration};

use criterion::{Criterion, criterion_group, criterion_main};
use faer::Col;
use shrinkage::{
    Gaussian, Hybrid, HybridStep, Normalization, Problem, ProximalGradient, Termination,
};

#[path = "common/slope.rs"]
mod fixture;

fn solves(c: &mut Criterion) {
    for (n, p) in [(256, 64), (512, 128)] {
        for (name, block_size) in [("distinct", 1), ("tied", 4), ("clustered", p)] {
            let (x, y, penalty) = fixture::dataset(n, p, block_size, 0.8);
            let problem = Problem::new(&x, Gaussian::new(&y).fit_intercept(false), penalty.clone())
                .normalize(Normalization::None);
            let pg = ProximalGradient::<Col<f64>>::new()
                .tolerance(1e-8)
                .max_iterations(100_000);
            let reference = problem.fit_with(&pg).unwrap();
            assert_eq!(reference.termination(), Termination::Converged);
            assert!(fixture::mapping(&x, &y, reference.coefficients(), &penalty) < 5e-8);
            let mut group = c.benchmark_group(format!("slope_{name}_{n}x{p}"));
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
                assert_eq!(fit.cluster_values().len(), p / block_size);
                assert!(
                    (fit.objective() - reference.objective()).abs()
                        < 1e-8 * (1.0 + reference.objective().abs())
                );
                assert!(fixture::mapping(&x, &y, fit.coefficients(), &penalty) < 5e-8);
                eprintln!(
                    "{name},{n},{p},{label},clusters={},pg_steps={},sweeps={},checks={},backtracks={}",
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
            for (label, budget) in [("setup_and_first_pg", 1), ("setup_sweeps_and_second_pg", 7)] {
                let solver = Hybrid::<Col<f64>>::new()
                    .tolerance(1e-14)
                    .max_iterations(budget);
                group.bench_function(label, |b| {
                    b.iter(|| black_box(&problem).fit_with(black_box(&solver)).unwrap())
                });
            }
            group.finish();
        }
    }
}

criterion_group!(benches, solves);
criterion_main!(benches);
