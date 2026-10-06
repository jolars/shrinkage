//! Fit-only allocation counts and peak live heap, separate from timing runs.

use std::hint::black_box;

use shrinkage::{Lasso, LassoFit, Termination};

#[path = "common/fixture.rs"]
mod fixture;

#[global_allocator]
static ALLOCATOR: dhat::Alloc = dhat::Alloc;

fn measure(name: &str, n: usize, p: usize, fit: impl FnOnce() -> LassoFit) {
    // A standalone harness prevents test-runner allocations from entering the
    // profile. Inputs already exist, so only fitting and its result are counted.
    let profiler = dhat::Profiler::builder().testing().build();
    let result = black_box(fit());
    let stats = dhat::HeapStats::get();
    let iterations = result.iterations();
    assert_eq!(result.termination(), Termination::Converged);

    // Leave room for linear workspaces while rejecting a dense design copy or
    // Gram matrix on the larger cases. This is a budget, not an exact ABI size.
    dhat::assert!(stats.max_bytes <= 128 * (n + p));
    dhat::assert!(stats.total_blocks > 0);
    drop(result);
    let after_drop = dhat::HeapStats::get();
    dhat::assert_eq!(after_drop.curr_bytes, 0);
    drop(profiler);

    println!(
        "{name},{n},{p},{iterations},{},{},{},{}",
        stats.total_blocks, stats.total_bytes, stats.max_bytes, stats.curr_bytes
    );
}

fn main() {
    println!("storage,n,p,iterations,allocations,allocated_bytes,peak_live_bytes,result_bytes");
    for (n, p) in [(1_024, 64), (4_096, 64), (1_024, 1_024)] {
        let (dense, sparse, y) = fixture::dataset(n, p);
        let model = Lasso::new(0.03);
        measure("dense", n, p, || model.fit(&dense, &y).unwrap());
        measure("csc", n, p, || model.fit(&sparse, &y).unwrap());
    }
}
