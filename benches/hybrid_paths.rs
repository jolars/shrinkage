//! Complete warm paths and cold fits at identical tuning points and accuracy.

use std::{hint::black_box, time::Instant};

use faer::{Col, Mat};
use shrinkage::{
    Binomial, ChainFusedLasso, ClusteredLasso, Gaussian, Hybrid, HybridFit, Normalization,
    PathPenalty, Problem, ProximalPenalty, Slope, SmoothDatafit, Termination,
};

#[path = "common/hybrid_irls.rs"]
mod fixture;

#[global_allocator]
static ALLOCATOR: dhat::Alloc = dhat::Alloc;

const TOLERANCE: f64 = 1e-8;

fn check<D: SmoothDatafit, P: ProximalPenalty>(
    x: &Mat<f64>,
    datafit: &D,
    penalty: &P,
    fit: &HybridFit,
) {
    assert_eq!(fit.termination(), Termination::Converged);
    let eta = fit.predict(x).unwrap();
    let mut derivative = vec![0.0; x.nrows()];
    datafit.gradient(&eta, &mut derivative).unwrap();
    let input: Vec<_> = (0..x.ncols())
        .map(|j| {
            fit.coefficients()[j]
                - fit.step_size()
                    * (0..x.nrows())
                        .map(|i| x[(i, j)] * derivative[i])
                        .sum::<f64>()
        })
        .collect();
    let mut prox = vec![0.0; x.ncols()];
    penalty.prox(&input, fit.step_size(), &mut prox).unwrap();
    let mapping = fit
        .coefficients()
        .iter()
        .zip(prox)
        .map(|(b, z)| (b - z).abs() / fit.step_size())
        .fold(derivative.iter().sum::<f64>().abs(), f64::max);
    assert!(mapping <= 1.01 * TOLERANCE);
    assert!(
        (fit.objective()
            - datafit.value(&eta).unwrap()
            - penalty.value(fit.coefficients()).unwrap())
        .abs()
            < 1e-12
    );
}

fn profile<T>(label: &str, mut run: impl FnMut() -> T, mut validate: impl FnMut(&T)) {
    let start = Instant::now();
    let result = black_box(run());
    let elapsed = start.elapsed();
    validate(&result);
    drop(result);
    // Measure heap work separately so allocation instrumentation does not
    // dominate the complete-solve timing comparison.
    let profiler = dhat::Profiler::builder().testing().build();
    let result = black_box(run());
    let stats = dhat::HeapStats::get();
    validate(&result);
    drop(result);
    let remaining = dhat::HeapStats::get();
    dhat::assert_eq!(remaining.curr_bytes, 0);
    drop(profiler);
    println!(
        "{label},{},{},{},{}",
        elapsed.as_micros(),
        stats.total_blocks,
        stats.total_bytes,
        stats.max_bytes
    );
}

fn main() {
    println!(
        "family,penalty,case,repeat,points,microseconds,allocations,allocated_bytes,peak_bytes"
    );
    let repeats = if std::env::args().any(|arg| arg == "--quick") {
        1
    } else {
        3
    };
    let (n, p) = (128, 16);
    let (x, binary) = fixture::dataset(n, p);
    let gaussian: Vec<_> = binary.iter().map(|&y| (y / (1.0 - y)).ln()).collect();
    let solver = Hybrid::<Col<f64>>::new()
        .tolerance(TOLERANCE)
        .max_iterations(100_000);
    macro_rules! benchmark {
        ($family:literal, $datafit:expr, $name:literal, $base:expr) => {{
            let datafit = $datafit;
            let base = $base;
            let points: Vec<_> = [8.0, 4.0, 2.0, 1.0, 0.5, 0.25]
                .iter()
                .map(|&lambda| base.at_strength(lambda).unwrap())
                .collect();
            let problem = Problem::new(&x, datafit, base).normalize(Normalization::None);
            for repeat in 0..repeats {
                let mut objectives = Vec::with_capacity(points.len());
                profile(
                    &format!("{},{},cold,{repeat},{}", $family, $name, points.len()),
                    || {
                        points
                            .iter()
                            .map(|penalty| {
                                Problem::new(&x, datafit, penalty.clone())
                                    .normalize(Normalization::None)
                                    .fit_with(&solver)
                                    .unwrap()
                            })
                            .collect::<Vec<_>>()
                    },
                    |fits| {
                        objectives.clear();
                        for (fit, penalty) in fits.iter().zip(&points) {
                            check(&x, &datafit, penalty, fit);
                            if $family == "binomial" {
                                fixture::check(&x, &binary, fit, penalty);
                            }
                            objectives.push(fit.objective());
                        }
                    },
                );
                profile(
                    &format!("{},{},warm,{repeat},{}", $family, $name, points.len()),
                    || problem.fit_path_with(&solver, points.clone()).unwrap(),
                    |path| {
                        for ((penalty, fit), objective) in path.iter().zip(&objectives) {
                            check(&x, &datafit, penalty, fit);
                            if $family == "binomial" {
                                fixture::check(&x, &binary, fit, penalty);
                            }
                            assert!((fit.objective() - objective).abs() < 1e-10);
                        }
                    },
                );
            }
        }};
    }
    benchmark!(
        "gaussian",
        Gaussian::new(&gaussian),
        "clustered",
        ClusteredLasso::new(0.005, 0.02 / p as f64)
    );
    benchmark!(
        "gaussian",
        Gaussian::new(&gaussian),
        "chain",
        ChainFusedLasso::new(0.005, 0.02)
    );
    benchmark!(
        "gaussian",
        Gaussian::new(&gaussian),
        "slope",
        Slope::new(
            (0..p)
                .map(|j| 0.005 + 0.02 * (p - j) as f64 / p as f64)
                .collect()
        )
    );
    benchmark!(
        "binomial",
        Binomial::new(&binary),
        "clustered",
        ClusteredLasso::new(0.005, 0.02 / p as f64)
    );
    benchmark!(
        "binomial",
        Binomial::new(&binary),
        "chain",
        ChainFusedLasso::new(0.005, 0.02)
    );
    benchmark!(
        "binomial",
        Binomial::new(&binary),
        "slope",
        Slope::new(
            (0..p)
                .map(|j| 0.005 + 0.02 * (p - j) as f64 / p as f64)
                .collect()
        )
    );
}
