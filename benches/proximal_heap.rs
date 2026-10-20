//! Isolated allocation measurements for normalized products, prox, and paths.

use faer::Col;
use shrinkage::lazymatrix::{LazyMatrix, MatTransposeVecInto, MatVecInto};
use shrinkage::{
    DisjointGroups, ElasticNet, Gaussian, GroupLasso, Masked, Penalty, Problem, ProximalGradient,
    ProximalPenalty, Scaled, Slope, Termination, UncoveredPolicy,
};
use std::hint::black_box;

#[path = "common/fixture.rs"]
mod fixture;

#[global_allocator]
static ALLOCATOR: dhat::Alloc = dhat::Alloc;

fn measure(name: &str, calls: usize, mut operation: impl FnMut(), expected_blocks: Option<u64>) {
    let profiler = dhat::Profiler::builder().testing().build();
    for _ in 0..calls {
        operation();
    }
    let stats = dhat::HeapStats::get();
    if let Some(expected) = expected_blocks {
        dhat::assert_eq!(stats.total_blocks, expected);
    }
    dhat::assert_eq!(stats.curr_bytes, 0);
    dhat::assert!(stats.max_bytes <= 128 * (1_024 + 64));
    drop(profiler);
    println!(
        "{name},{calls},{},{},{}",
        stats.total_blocks, stats.total_bytes, stats.max_bytes
    );
}

fn main() {
    let (dense, sparse, y) = fixture::dataset(1_024, 64);
    let input = Col::from_fn(64, |j| j as f64 / 64.0);
    let mut output = Col::zeros(1_024);
    let mut transpose = Col::zeros(64);
    let mut scratch = Col::zeros(64);
    let input_slice = vec![0.5; 64];
    let mut prox_output = vec![0.0; 64];
    let penalty = ElasticNet::new(0.03, 0.1);
    let normalized_dense = LazyMatrix::from_parts(&dense, Some(vec![0.1; 64]), Some(vec![2.0; 64]));
    let normalized_sparse =
        LazyMatrix::from_parts(&sparse, Some(vec![0.1; 64]), Some(vec![2.0; 64]));
    println!("operation,calls,allocations,allocated_bytes,peak_live_bytes");
    measure(
        "dense_forward_legacy",
        100,
        || {
            normalized_dense
                .matvec_into(black_box(&input), &mut output)
                .unwrap()
        },
        Some(100),
    );
    measure(
        "csc_forward_legacy",
        100,
        || {
            normalized_sparse
                .matvec_into(black_box(&input), &mut output)
                .unwrap()
        },
        Some(100),
    );
    measure(
        "dense_transpose",
        100,
        || {
            normalized_dense
                .mat_transpose_vec_into(black_box(&output), &mut transpose)
                .unwrap()
        },
        Some(0),
    );
    measure(
        "csc_transpose",
        100,
        || {
            normalized_sparse
                .mat_transpose_vec_into(black_box(&output), &mut transpose)
                .unwrap()
        },
        Some(0),
    );
    measure(
        "dense_forward_upstream_scratch",
        100,
        || {
            normalized_dense
                .matvec_scaled_with_workspace(
                    1.0,
                    black_box(&input),
                    0.0,
                    &mut output,
                    &mut scratch,
                )
                .unwrap()
        },
        Some(0),
    );
    measure(
        "csc_forward_upstream_scratch",
        100,
        || {
            normalized_sparse
                .matvec_scaled_with_workspace(
                    1.0,
                    black_box(&input),
                    0.0,
                    &mut output,
                    &mut scratch,
                )
                .unwrap()
        },
        Some(0),
    );
    measure(
        "elastic_net_prox",
        100,
        || {
            penalty
                .prox(black_box(&input_slice), 0.5, &mut prox_output)
                .unwrap()
        },
        Some(0),
    );
    let slope = Slope::new((0..64).map(|j| (64 - j) as f64 / 64.0).collect());
    let groups = DisjointGroups::new(
        64,
        (0..8).map(|g| (8 * g..8 * (g + 1)).collect()).collect(),
        vec![1.0; 8],
        UncoveredPolicy::Reject,
    )
    .unwrap();
    let group_lasso = GroupLasso::new(0.03, groups);
    measure(
        "group_lasso_prox",
        100,
        || {
            group_lasso
                .prox(black_box(&input_slice), 0.5, &mut prox_output)
                .unwrap();
        },
        Some(0),
    );
    measure(
        "group_lasso_value",
        100,
        || {
            black_box(group_lasso.value(black_box(&input_slice)).unwrap());
        },
        Some(0),
    );
    let masked = Masked::new((0..64).map(|j| j % 2 == 0).collect(), penalty);
    measure(
        "masked_elastic_net_prox",
        100,
        || {
            masked
                .prox(black_box(&input_slice), 0.5, &mut prox_output)
                .unwrap()
        },
        Some(0),
    );
    measure(
        "masked_elastic_net_value",
        100,
        || {
            black_box(masked.value(black_box(&input_slice)).unwrap());
        },
        Some(0),
    );
    measure(
        "slope_prox",
        100,
        || {
            slope
                .prox(black_box(&input_slice), 0.5, &mut prox_output)
                .unwrap();
        },
        Some(0),
    );
    measure(
        "slope_value",
        100,
        || {
            black_box(slope.value(black_box(&input_slice)).unwrap());
        },
        Some(0),
    );
    for scale in [0.0, 0.25] {
        let scaled = Scaled::new(scale, penalty);
        measure(
            &format!("scaled_elastic_net_prox_{scale}"),
            100,
            || {
                scaled
                    .prox(black_box(&input_slice), 0.5, &mut prox_output)
                    .unwrap()
            },
            Some(0),
        );
        measure(
            &format!("scaled_elastic_net_value_{scale}"),
            100,
            || {
                black_box(scaled.value(black_box(&input_slice)).unwrap());
            },
            Some(0),
        );
        let scaled = Scaled::new(scale, Slope::new(slope.weights().to_vec()));
        measure(
            &format!("scaled_slope_prox_{scale}"),
            100,
            || {
                scaled
                    .prox(black_box(&input_slice), 0.5, &mut prox_output)
                    .unwrap()
            },
            Some(0),
        );
        measure(
            &format!("scaled_slope_value_{scale}"),
            100,
            || {
                black_box(scaled.value(black_box(&input_slice)).unwrap());
            },
            Some(0),
        );
    }
    let solver = ProximalGradient::<Col<f64>>::new();
    let problem = Problem::new(&dense, Gaussian::new(&y), penalty);
    let penalties = [
        ElasticNet::new(0.1, 0.1),
        penalty,
        ElasticNet::new(0.01, 0.1),
    ];
    measure(
        "dense_path",
        1,
        || {
            let fits = solver.solve_path(&problem, black_box(&penalties)).unwrap();
            assert!(
                fits.iter()
                    .all(|fit| fit.termination() == Termination::Converged)
            );
            black_box(fits);
        },
        None,
    );
    measure(
        "dense_cold_points",
        1,
        || {
            for penalty in penalties {
                let _ = black_box(
                    Problem::new(&dense, Gaussian::new(&y), penalty)
                        .fit_with(&solver)
                        .unwrap(),
                );
            }
        },
        None,
    );
    let sparse_problem = Problem::new(&sparse, Gaussian::new(&y), penalty);
    measure(
        "csc_path",
        1,
        || {
            let fits = solver
                .solve_path(&sparse_problem, black_box(&penalties))
                .unwrap();
            assert!(
                fits.iter()
                    .all(|fit| fit.termination() == Termination::Converged)
            );
            black_box(fits);
        },
        None,
    );
    measure(
        "csc_cold_points",
        1,
        || {
            for penalty in penalties {
                let _ = black_box(
                    Problem::new(&sparse, Gaussian::new(&y), penalty)
                        .fit_with(&solver)
                        .unwrap(),
                );
            }
        },
        None,
    );
}
