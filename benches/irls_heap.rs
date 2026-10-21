//! Fit-only IRLS workspace and allocation measurements.

use shrinkage::{Binomial, ElasticNet, Irls, IrlsFit, Problem, Termination};
use std::hint::black_box;

#[path = "common/binomial.rs"]
mod binomial;
#[path = "common/fixture.rs"]
mod fixture;

#[global_allocator]
static ALLOCATOR: dhat::Alloc = dhat::Alloc;

fn measure(
    name: &str,
    n: usize,
    p: usize,
    converged: bool,
    fit: impl FnOnce() -> IrlsFit,
) -> usize {
    let profiler = dhat::Profiler::builder().testing().build();
    let fit = black_box(fit());
    let stats = dhat::HeapStats::get();
    if converged {
        assert_eq!(fit.termination(), Termination::Converged);
    }
    // Allow reusable linear workspaces, but reject normalized design or Gram copies.
    dhat::assert!(stats.max_bytes <= 256 * (n + p));
    let iterations = fit.iterations();
    let sweeps = fit.coordinate_sweeps();
    drop(fit);
    let after_drop = dhat::HeapStats::get();
    dhat::assert_eq!(after_drop.curr_bytes, 0);
    drop(profiler);
    println!(
        "{name},{n},{p},{iterations},{sweeps},{},{},{}",
        stats.total_blocks, stats.total_bytes, stats.max_bytes
    );
    stats.total_blocks as usize
}

fn main() {
    println!(
        "storage,n,p,outer_steps,coordinate_sweeps,allocations,allocated_bytes,peak_live_bytes"
    );
    for (n, p) in [(1_024, 64), (4_096, 64), (1_024, 1_024)] {
        let (dense, sparse, latent) = fixture::dataset(n, p);
        let y = binomial::responses(&latent);
        let penalty = ElasticNet::new(0.03, 0.1);
        let model = Problem::new(&dense, Binomial::new(&y), penalty);
        let first = measure("dense_one_step", n, p, false, || {
            model.fit_with(&Irls::new().max_iterations(1)).unwrap()
        });
        let complete = measure("dense", n, p, true, || {
            model.fit_with(&Irls::new()).unwrap()
        });
        assert_eq!(
            first, complete,
            "outer iterations must reuse their workspaces"
        );
        let model = Problem::new(&sparse, Binomial::new(&y), penalty);
        let first = measure("csc_one_step", n, p, false, || {
            model.fit_with(&Irls::new().max_iterations(1)).unwrap()
        });
        let complete = measure("csc", n, p, true, || model.fit_with(&Irls::new()).unwrap());
        assert_eq!(
            first, complete,
            "coordinate sweeps must reuse their workspaces"
        );
    }
}
