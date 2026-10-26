//! Warmed proximal buffers and complete-solve aggregate storage.

use std::hint::black_box;

use faer::Col;
use shrinkage::{
    Gaussian, Hybrid, Normalization, Penalty, Problem, ProximalPenalty, Slope, Termination,
};

#[path = "common/slope.rs"]
mod fixture;

#[global_allocator]
static ALLOCATOR: dhat::Alloc = dhat::Alloc;

fn main() {
    let input: Vec<_> = (0..1_024).map(|j| (j % 37) as f64 - 18.0).collect();
    let penalty = Slope::new(
        (0..input.len())
            .map(|j| 1.0 - j as f64 / input.len() as f64)
            .collect(),
    );
    let mut output = vec![0.0; input.len()];
    penalty.prox(&input, 0.5, &mut output).unwrap();
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
    for (n, p) in [(128, 64), (512, 64), (512, 256)] {
        for (name, block_size, correlation, maximum_clusters) in [
            ("clustered", p, 0.0, 1),
            ("distinct", 1, 0.0, p),
            ("changing", p, 0.8, p),
        ] {
            let (x, y, penalty) = fixture::dataset(n, p, block_size, correlation);
            let problem = Problem::new(&x, Gaussian::new(&y).fit_intercept(false), penalty.clone())
                .normalize(Normalization::None);
            let profiler = dhat::Profiler::builder().testing().build();
            let fit = problem
                .fit_with(
                    &Hybrid::<Col<f64>>::new()
                        .tolerance(1e-8)
                        .max_iterations(100_000),
                )
                .unwrap();
            let stats = dhat::HeapStats::get();
            dhat::assert!(stats.max_bytes <= 256 * (n * maximum_clusters + n + p));
            assert_eq!(fit.termination(), Termination::Converged);
            assert_eq!(fit.cluster_values().len(), p / block_size);
            assert!(fixture::mapping(&x, &y, fit.coefficients(), &penalty) < 5e-8);
            let clusters = fit.cluster_values().len();
            let pg_steps = fit.pg_steps();
            let sweeps = fit.coordinate_sweeps();
            drop(fit);
            drop(problem);
            drop(penalty);
            let after_drop = dhat::HeapStats::get();
            dhat::assert_eq!(after_drop.curr_bytes, 0);
            drop(profiler);
            println!(
                "{name},{n},{p},{clusters},{pg_steps},{sweeps},{},{},{}",
                stats.total_blocks, stats.total_bytes, stats.max_bytes
            );
        }
    }
}
