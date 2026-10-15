//! Value and proximal costs with explicit coefficient exclusions.

use std::{hint::black_box, time::Duration};

use criterion::{Criterion, criterion_group, criterion_main};
use shrinkage::{ElasticNet, Masked, Penalty, ProximalPenalty};

fn benchmark_masked(c: &mut Criterion) {
    for length in [64, 1_024, 16_384] {
        let input: Vec<_> = (0..length)
            .map(|i| (i * 37 % length) as f64 / length as f64 - 0.5)
            .collect();
        let mut output = vec![0.0; length];
        let inner = ElasticNet::new(0.1, 0.2);
        let mut group = c.benchmark_group(format!("masked/elastic_net/{length}"));
        group.sample_size(20);
        group.warm_up_time(Duration::from_secs(1));
        group.measurement_time(Duration::from_secs(2));
        group.bench_function("unmasked_value", |b| {
            b.iter(|| inner.value(black_box(&input)).unwrap());
        });
        group.bench_function("unmasked_prox", |b| {
            b.iter(|| {
                inner
                    .prox(black_box(&input), 0.5, black_box(&mut output))
                    .unwrap();
            });
        });
        for (name, mask) in [
            ("none_excluded", vec![false; length]),
            ("half_excluded", (0..length).map(|j| j % 2 == 0).collect()),
            ("all_excluded", vec![true; length]),
        ] {
            let penalty = Masked::new(mask, inner);
            group.bench_function(format!("{name}_value"), |b| {
                b.iter(|| penalty.value(black_box(&input)).unwrap());
            });
            group.bench_function(format!("{name}_prox"), |b| {
                b.iter(|| {
                    penalty
                        .prox(black_box(&input), 0.5, black_box(&mut output))
                        .unwrap();
                });
            });
        }
        group.finish();
    }
}

criterion_group!(benches, benchmark_masked);
criterion_main!(benches);
