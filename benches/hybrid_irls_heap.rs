//! Weighted aggregate storage across repeated logistic quadratic models.

use faer::Col;
use shrinkage::{
    Binomial, ChainFusedLasso, ClusteredLasso, Hybrid, Normalization, Problem, Slope, Termination,
};

#[path = "common/hybrid_irls.rs"]
mod fixture;

#[global_allocator]
static ALLOCATOR: dhat::Alloc = dhat::Alloc;

fn main() {
    println!("penalty,n,p,clusters,outer,pg,sweeps,allocations,allocated_bytes,peak_live_bytes");
    for (n, p) in [(128, 16), (512, 16), (128, 64)] {
        let (x, y) = fixture::dataset(n, p);
        macro_rules! profile {
            ($name:literal, $penalty:expr) => {{
                let penalty = $penalty;
                let problem = Problem::new(&x, Binomial::new(&y), penalty.clone())
                    .normalize(Normalization::None);
                let profiler = dhat::Profiler::builder().testing().build();
                let fit = problem
                    .fit_with(
                        &Hybrid::<Col<f64>>::new()
                            .tolerance(1e-7)
                            .max_iterations(100_000),
                    )
                    .unwrap();
                let stats = dhat::HeapStats::get();
                assert_eq!(fit.termination(), Termination::Converged);
                assert!(fit.outer_iterations() > 1);
                dhat::assert!(stats.max_bytes <= 256 * (n * p + n + p));
                let counts = (
                    fit.cluster_values().len(),
                    fit.outer_iterations(),
                    fit.pg_steps(),
                    fit.coordinate_sweeps(),
                );
                fixture::check(&x, &y, &fit, &penalty);
                drop(fit);
                drop(problem);
                drop(penalty);
                let remaining = dhat::HeapStats::get();
                dhat::assert_eq!(remaining.curr_bytes, 0);
                drop(profiler);
                println!(
                    "{},{n},{p},{},{},{},{},{},{},{}",
                    $name,
                    counts.0,
                    counts.1,
                    counts.2,
                    counts.3,
                    stats.total_blocks,
                    stats.total_bytes,
                    stats.max_bytes
                );
            }};
        }
        profile!("clustered", ClusteredLasso::new(0.001, 0.02 / p as f64));
        profile!("chain", ChainFusedLasso::new(0.001, 0.02));
        profile!(
            "slope",
            Slope::new(
                (0..p)
                    .map(|j| 0.001 + 0.02 * (p - j) as f64 / p as f64)
                    .collect()
            )
        );
    }
}
