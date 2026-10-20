//! Weighted block operations with caller-owned output and no scratch allocation.

use std::{hint::black_box, time::Duration};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use shrinkage::{DisjointGroups, GroupLasso, Penalty, ProximalPenalty, UncoveredPolicy};

fn benchmark_groups(c: &mut Criterion) {
    let mut group = c.benchmark_group("group_lasso");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));
    for length in [64, 1_024] {
        let input: Vec<_> = (0..length)
            .map(|j| ((j * 37) % 29) as f64 / 7.0 - 2.0)
            .collect();
        let mut output = vec![0.0; length];
        for size in [1, 8, 64] {
            let groups = DisjointGroups::new(
                length,
                (0..length / size)
                    .map(|g| (g * size..(g + 1) * size).collect())
                    .collect(),
                vec![1.5; length / size],
                UncoveredPolicy::Reject,
            )
            .unwrap();
            let penalty = GroupLasso::new(0.3, groups);
            group.bench_function(
                BenchmarkId::new(format!("prox_block_{size}"), length),
                |b| {
                    b.iter(|| {
                        penalty
                            .prox(black_box(&input), black_box(0.5), black_box(&mut output))
                            .unwrap()
                    });
                },
            );
            group.bench_function(
                BenchmarkId::new(format!("value_block_{size}"), length),
                |b| {
                    b.iter(|| penalty.value(black_box(&input)).unwrap());
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, benchmark_groups);
criterion_main!(benches);
