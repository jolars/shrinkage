//! Native buffer construction and copies, measured separately from products.

use std::{hint::black_box, time::Duration};

use criterion::{Criterion, criterion_group, criterion_main};
use faer::Col;
use shrinkage::lazymatrix::{
    ColumnStats, LazyMatrix, MatTransposeVecInto, MatVecInto, Normalization as MatrixNormalization,
    RawColumns,
};
use shrinkage::{MatrixDesign, Normalization, ProximalVector, RuntimeDesign};

#[path = "common/fixture.rs"]
mod fixture;

fn copies<V: ProximalVector>(c: &mut Criterion, backend: &str, n: usize, p: usize) {
    let parameters: Vec<_> = (0..p).map(|j| j as f64 / p as f64).collect();
    let derivative: Vec<_> = (0..n).map(|i| (i % 7) as f64 - 3.0).collect();
    let mut native_parameters = V::zeros(p);
    let mut native_derivative = V::zeros(n);
    for (j, &value) in parameters.iter().enumerate() {
        native_parameters.set(j, value);
    }
    for (i, &value) in derivative.iter().enumerate() {
        native_derivative.set(i, value);
    }
    let mut parameters_output = vec![0.0; p];
    let mut derivative_output = vec![0.0; n];
    let mut group = c.benchmark_group(format!("proximal_buffers_{backend}_{n}x{p}"));
    group.sample_size(30);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(1));
    group.bench_function("construct", |b| {
        b.iter(|| black_box((V::zeros(black_box(p)), V::zeros(black_box(n)))))
    });
    group.bench_function("into_native", |b| {
        b.iter(|| {
            for (j, &value) in black_box(&parameters).iter().enumerate() {
                native_parameters.set(j, value);
            }
            for (i, &value) in black_box(&derivative).iter().enumerate() {
                native_derivative.set(i, value);
            }
            black_box((&native_parameters, &native_derivative));
        })
    });
    group.bench_function("from_native", |b| {
        b.iter(|| {
            let (parameters, derivative) = black_box((&native_parameters, &native_derivative));
            for (j, value) in parameters_output.iter_mut().enumerate() {
                *value = parameters.get(j);
            }
            for (i, value) in derivative_output.iter_mut().enumerate() {
                *value = derivative.get(i);
            }
            black_box((&parameters_output, &derivative_output));
        })
    });
    group.finish();
}

fn products<M>(c: &mut Criterion, storage: &str, matrix: &M, n: usize, p: usize)
where
    M: RawColumns<f64> + ColumnStats<f64> + MatVecInto<Col<f64>> + MatTransposeVecInto<Col<f64>>,
    M::Error: std::error::Error + 'static,
{
    let parameters: Vec<_> = (0..p).map(|j| j as f64 / p as f64).collect();
    let derivative: Vec<_> = (0..n).map(|i| (i % 7) as f64 - 3.0).collect();
    let native_parameters = Col::from_fn(p, |j| parameters[j]);
    let native_derivative = Col::from_fn(n, |i| derivative[i]);
    let mut native_predictor = Col::zeros(n);
    let mut native_gradient = Col::zeros(p);
    let mut predictor = vec![f64::NAN; n];
    let mut gradient = vec![f64::NAN; p];
    let normalized = LazyMatrix::new(matrix, MatrixNormalization::standardize()).unwrap();
    let design = MatrixDesign::<_, Col<f64>>::new(matrix);
    let mut prepared = design
        .prepare(Normalization::Standardize, None, None, false)
        .unwrap();
    prepared.forward(&parameters, &mut predictor).unwrap();
    prepared.transpose(&derivative, &mut gradient).unwrap();
    normalized
        .matvec_into(&native_parameters, &mut native_predictor)
        .unwrap();
    normalized
        .mat_transpose_vec_into(&native_derivative, &mut native_gradient)
        .unwrap();
    for (i, &value) in predictor.iter().enumerate() {
        assert!((value - native_predictor[i]).abs() <= 1e-12 * (1.0 + value.abs()));
    }
    for (j, &value) in gradient.iter().enumerate() {
        assert!((value - native_gradient[j]).abs() <= 1e-12 * (1.0 + value.abs()));
    }
    let mut group = c.benchmark_group(format!("proximal_products_{storage}_{n}x{p}"));
    group.sample_size(30);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(1));
    group.bench_function("raw_forward", |b| {
        b.iter(|| {
            matrix
                .matvec_into(
                    black_box(&native_parameters),
                    black_box(&mut native_predictor),
                )
                .unwrap()
        })
    });
    group.bench_function("normalized_forward", |b| {
        b.iter(|| {
            normalized
                .matvec_into(
                    black_box(&native_parameters),
                    black_box(&mut native_predictor),
                )
                .unwrap()
        })
    });
    group.bench_function("consumer_forward", |b| {
        b.iter(|| {
            prepared
                .forward(black_box(&parameters), black_box(&mut predictor))
                .unwrap()
        })
    });
    group.bench_function("raw_transpose", |b| {
        b.iter(|| {
            matrix
                .mat_transpose_vec_into(
                    black_box(&native_derivative),
                    black_box(&mut native_gradient),
                )
                .unwrap()
        })
    });
    group.bench_function("normalized_transpose", |b| {
        b.iter(|| {
            normalized
                .mat_transpose_vec_into(
                    black_box(&native_derivative),
                    black_box(&mut native_gradient),
                )
                .unwrap()
        })
    });
    group.bench_function("consumer_transpose", |b| {
        b.iter(|| {
            prepared
                .transpose(black_box(&derivative), black_box(&mut gradient))
                .unwrap()
        })
    });
    group.finish();
}

fn benchmark(c: &mut Criterion) {
    for (n, p) in [(1_024, 64), (16_384, 16)] {
        copies::<Vec<f64>>(c, "vec", n, p);
        copies::<Col<f64>>(c, "faer", n, p);
        #[cfg(feature = "nalgebra_v0_34")]
        copies::<nalgebra::DVector<f64>>(c, "nalgebra", n, p);
        #[cfg(feature = "ndarray_v0_17")]
        copies::<ndarray::Array1<f64>>(c, "ndarray", n, p);
        let (dense, sparse, _) = fixture::dataset(n, p);
        products(c, "dense", &dense, n, p);
        products(c, "csc", &sparse, n, p);
    }
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
