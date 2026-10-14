//! Isolated exact SLOPE proximal maps with reusable sorting and pooling storage.

use std::{hint::black_box, time::Duration};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use shrinkage::{ProximalPenalty, Slope};

fn benchmark_slope(c: &mut Criterion) {
    let mut group = c.benchmark_group("slope_prox");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));
    for length in [64, 1_024, 16_384] {
        let penalty = Slope::new(
            (0..length)
                .map(|i| 2.0 * (1.0 - i as f64 / length as f64))
                .collect(),
        );
        let mut output = vec![0.0; length];
        for pattern in ["distinct", "ties"] {
            let input: Vec<_> = (0..length)
                .map(|i| {
                    let magnitude = if pattern == "ties" {
                        (i * 37 % 8) as f64 / 2.0
                    } else {
                        (i * 37 % length) as f64 * 4.0 / length as f64
                    };
                    if i % 2 == 0 { magnitude } else { -magnitude }
                })
                .collect();
            penalty.prox(&input, 0.5, &mut output).unwrap();
            group.bench_with_input(BenchmarkId::new(pattern, length), &input, |b, input| {
                b.iter(|| {
                    penalty
                        .prox(black_box(input), black_box(0.5), black_box(&mut output))
                        .unwrap();
                });
            });
        }
    }
    group.finish();
}

criterion_group!(benches, benchmark_slope);
criterion_main!(benches);
