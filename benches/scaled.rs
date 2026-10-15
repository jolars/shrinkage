//! Complete value and proximal scaling compared with built-in strengths.

use std::{hint::black_box, time::Duration};

use criterion::{Criterion, criterion_group, criterion_main};
use shrinkage::{ElasticNet, Penalty, ProximalPenalty, Scaled, Slope};

fn benchmark_penalty<P: ProximalPenalty>(
    c: &mut Criterion,
    name: &str,
    scale: f64,
    inner: P,
    direct: P,
    input: &[f64],
) {
    let scaled = Scaled::new(scale, inner);
    let mut output = vec![0.0; input.len()];
    let mut group = c.benchmark_group(format!("scaled/{name}/{}/{scale}", input.len()));
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));
    group.bench_function("wrapped_value", |b| {
        b.iter(|| scaled.value(black_box(input)).unwrap());
    });
    group.bench_function("direct_value", |b| {
        b.iter(|| direct.value(black_box(input)).unwrap());
    });
    group.bench_function("wrapped_prox", |b| {
        b.iter(|| {
            scaled
                .prox(black_box(input), black_box(0.5), black_box(&mut output))
                .unwrap();
        });
    });
    group.bench_function("direct_prox", |b| {
        b.iter(|| {
            direct
                .prox(black_box(input), black_box(0.5), black_box(&mut output))
                .unwrap();
        });
    });
    group.finish();
}

fn benchmark_scaled(c: &mut Criterion) {
    for length in [64, 1_024, 16_384] {
        let input: Vec<_> = (0..length)
            .map(|i| (i * 37 % length) as f64 / length as f64 - 0.5)
            .collect();
        let weights: Vec<_> = (0..length)
            .map(|i| 1.0 - i as f64 / length as f64)
            .collect();
        for scale in [0.0, 0.25] {
            benchmark_penalty(
                c,
                "elastic_net",
                scale,
                ElasticNet::new(0.1, 0.2),
                ElasticNet::new(scale * 0.1, scale * 0.2),
                &input,
            );
            benchmark_penalty(
                c,
                "slope",
                scale,
                Slope::new(weights.clone()),
                Slope::new(weights.iter().map(|weight| scale * weight).collect()),
                &input,
            );
        }
    }
}

criterion_group!(benches, benchmark_scaled);
criterion_main!(benches);
