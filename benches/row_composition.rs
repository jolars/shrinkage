//! Row gather/scatter costs for composed penalties versus a flat equivalent.

use std::{hint::black_box, time::Duration};

use criterion::{Criterion, criterion_group, criterion_main};
use shrinkage::{
    AxisPenalty, CoefficientLayout, ElasticNet, Masked, Penalty, PenaltyAxis, ProximalPenalty,
    Scaled,
};

fn benchmark_rows(c: &mut Criterion) {
    for (nfeatures, nresponses) in [(64, 4), (1_024, 4), (1_024, 32)] {
        let layout = CoefficientLayout::new(nfeatures, nresponses).unwrap();
        let input: Vec<_> = (0..layout.len())
            .map(|i| (i * 37 % 101) as f64 / 101.0 - 0.5)
            .collect();
        let mask: Vec<_> = (0..nresponses).map(|response| response % 2 == 0).collect();
        let row_penalty = AxisPenalty::new(
            layout,
            PenaltyAxis::Responses,
            Scaled::new(0.5, Masked::new(mask.clone(), ElasticNet::new(0.1, 0.2))),
        );
        let flat_mask = mask
            .iter()
            .flat_map(|&excluded| std::iter::repeat_n(excluded, nfeatures))
            .collect();
        let flat_penalty = Masked::new(flat_mask, ElasticNet::new(0.05, 0.1));
        let mut output = vec![0.0; layout.len()];
        let mut expected = vec![0.0; layout.len()];
        row_penalty.prox(&input, 0.5, &mut output).unwrap();
        flat_penalty.prox(&input, 0.5, &mut expected).unwrap();
        assert_eq!(output, expected);
        let row_value = row_penalty.value(&input).unwrap();
        let flat_value = flat_penalty.value(&input).unwrap();
        assert!((row_value - flat_value).abs() <= 1e-10 * (1.0 + flat_value));

        let mut group = c.benchmark_group(format!("row_composition/{nfeatures}x{nresponses}"));
        group.sample_size(20);
        group.warm_up_time(Duration::from_secs(1));
        group.measurement_time(Duration::from_secs(2));
        group.bench_function("rows_value", |b| {
            b.iter(|| row_penalty.value(black_box(&input)).unwrap());
        });
        group.bench_function("flat_value", |b| {
            b.iter(|| flat_penalty.value(black_box(&input)).unwrap());
        });
        group.bench_function("rows_prox", |b| {
            b.iter(|| {
                row_penalty
                    .prox(black_box(&input), 0.5, black_box(&mut output))
                    .unwrap()
            });
        });
        group.bench_function("flat_prox", |b| {
            b.iter(|| {
                flat_penalty
                    .prox(black_box(&input), 0.5, black_box(&mut output))
                    .unwrap()
            });
        });
        group.finish();
    }
}

criterion_group!(benches, benchmark_rows);
criterion_main!(benches);
