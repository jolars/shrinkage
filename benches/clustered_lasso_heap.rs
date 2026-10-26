//! Peak storage and warmed proximal allocations, excluding the caller's design.

use std::hint::black_box;

use faer::Col;
use shrinkage::{
    ClusteredLasso, Gaussian, Hybrid, Normalization, Penalty, Problem, ProximalPenalty, Termination,
};

#[path = "common/clustered.rs"]
mod fixture;

#[global_allocator]
static ALLOCATOR: dhat::Alloc = dhat::Alloc;

fn main() {
    let penalty = ClusteredLasso::new(0.1, 0.2);
    let input: Vec<_> = (0..1_024).map(|j| (j % 37) as f64 - 18.0).collect();
    let mut output = vec![0.0; input.len()];
    penalty.prox(&input, 0.5, &mut output).unwrap();
    penalty.value(&input).unwrap();
    let profiler = dhat::Profiler::builder().testing().build();
    for _ in 0..100 {
        penalty
            .prox(black_box(&input), 0.5, black_box(&mut output))
            .unwrap();
        black_box(penalty.value(black_box(&input)).unwrap());
    }
    let warmed = dhat::HeapStats::get();
    dhat::assert_eq!(warmed.total_blocks, 0);
    drop(profiler);
    println!("case,n,p,clusters,pg_steps,sweeps,allocations,allocated_bytes,peak_live_bytes");
    for (n, p) in [(128, 64), (512, 64), (128, 256)] {
        let (x, y) = fixture::dataset(n, p);
        for (label, fusion, budget) in [("fused", 1_000.0, 10_000), ("distinct", 0.0, 7)] {
            let penalty = ClusteredLasso::new(0.0, fusion);
            let problem = Problem::new(&x, Gaussian::new(&y).fit_intercept(false), penalty.clone())
                .normalize(Normalization::None);
            let profiler = dhat::Profiler::builder().testing().build();
            let fit = problem
                .fit_with(&Hybrid::<Col<f64>>::new().max_iterations(budget))
                .unwrap();
            let stats = dhat::HeapStats::get();
            let clusters = fit.cluster_values().len();
            // Both staged aggregate sets are dense. In the weak-fusion case
            // their storage approaches a second and third copy of the design.
            let maximum_clusters = if label == "fused" { 1 } else { p };
            dhat::assert!(stats.max_bytes <= 128 * (n * maximum_clusters + n + p));
            if label == "fused" {
                assert_eq!(fit.termination(), Termination::Converged);
                assert_eq!(clusters, 1);
                assert!(fixture::mapping(&x, &y, fit.coefficients(), &penalty) < 1e-5);
            }
            let pg_steps = fit.pg_steps();
            let sweeps = fit.coordinate_sweeps();
            drop(fit);
            drop(problem);
            drop(penalty);
            let after_drop = dhat::HeapStats::get();
            dhat::assert_eq!(after_drop.curr_bytes, 0);
            drop(profiler);
            println!(
                "{label},{n},{p},{clusters},{pg_steps},{sweeps},{},{},{}",
                stats.total_blocks, stats.total_bytes, stats.max_bytes
            );
        }
    }
}
