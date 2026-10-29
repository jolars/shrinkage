//! Complete dynamic-screening solves at matched original-space accuracy.
//!
//! Run with `--quick` for one timed solve per configuration; the default uses
//! five. Timing excludes the separate heap-profiling pass and includes design
//! preparation, certificates, restriction rebuilds, and final diagnostics.

use std::{hint::black_box, time::Instant};

use faer::{
    Col, Mat,
    sparse::{SparseColMat, Triplet},
};
use shrinkage::{
    Binomial, ChainFusedLasso, ClusteredLasso, Gaussian, Hybrid, HybridFit, Normalization, Problem,
    ProximalFit, ProximalGradient, ProximalPenalty, ScreeningDiagnostics, Slope, SmoothDatafit,
    Termination,
};

#[global_allocator]
static ALLOCATOR: dhat::Alloc = dhat::Alloc;

const TOLERANCE: f64 = 1e-7;

trait FitSummary {
    fn coefficients(&self) -> &[f64];
    fn intercept(&self) -> f64;
    fn step_size(&self) -> f64;
    fn objective(&self) -> f64;
    fn stopping_value(&self) -> f64;
    fn iterations(&self) -> usize;
    fn termination(&self) -> Termination;
    fn screening_diagnostics(&self) -> &ScreeningDiagnostics;
}

macro_rules! fit_summary {
    ($kind:ty) => {
        impl FitSummary for $kind {
            fn coefficients(&self) -> &[f64] {
                self.coefficients()
            }
            fn intercept(&self) -> f64 {
                self.intercept()
            }
            fn step_size(&self) -> f64 {
                self.step_size()
            }
            fn objective(&self) -> f64 {
                self.objective()
            }
            fn stopping_value(&self) -> f64 {
                self.stopping_value()
            }
            fn iterations(&self) -> usize {
                self.iterations()
            }
            fn termination(&self) -> Termination {
                self.termination()
            }
            fn screening_diagnostics(&self) -> &ScreeningDiagnostics {
                self.screening_diagnostics()
            }
        }
    };
}

fit_summary!(ProximalFit);
fit_summary!(HybridFit);

fn dataset(
    n: usize,
    p: usize,
    sparse_solution: bool,
) -> (Mat<f64>, SparseColMat<usize, f64>, Vec<f64>, Vec<f64>) {
    let mut seed = 20261029_u64;
    let x = Mat::from_fn(n, p, |_, _| {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let value = (seed >> 11) as f64 / (1_u64 << 53) as f64;
        if value < 0.12 {
            5.0 * (value / 0.12 - 0.5)
        } else {
            0.0
        }
    });
    let mut triplets = Vec::new();
    for j in 0..p {
        for i in 0..n {
            if x[(i, j)] != 0.0 {
                triplets.push(Triplet::new(i, j, x[(i, j)]));
            }
        }
    }
    let csc = SparseColMat::try_new_from_triplets(n, p, &triplets).unwrap();
    let eta: Vec<_> = (0..n)
        .map(|i| {
            0.3 + (0..p)
                .map(|j| {
                    let coefficient = if sparse_solution {
                        match j {
                            0 => 1.2,
                            1 => -0.8,
                            2 => 0.5,
                            _ => 0.0,
                        }
                    } else {
                        0.1 * ((j % 7) as f64 - 3.0)
                    };
                    x[(i, j)] * coefficient
                })
                .sum::<f64>()
        })
        .collect();
    let gaussian = eta
        .iter()
        .enumerate()
        .map(|(i, &eta)| eta + 0.002 * ((i * 7 % 11) as f64 - 5.0))
        .collect();
    let binomial = eta
        .into_iter()
        .map(|eta| 1.0 / (1.0 + (-eta).exp()))
        .collect();
    (x, csc, gaussian, binomial)
}

fn validate<D: SmoothDatafit, P: ProximalPenalty>(
    x: &Mat<f64>,
    datafit: &D,
    penalty: &P,
    fit: &impl FitSummary,
) {
    assert_eq!(fit.termination(), Termination::Converged);
    assert!(fit.stopping_value() <= TOLERANCE);
    let eta: Vec<_> = (0..x.nrows())
        .map(|i| {
            fit.intercept()
                + fit
                    .coefficients()
                    .iter()
                    .enumerate()
                    .map(|(j, &b)| x[(i, j)] * b)
                    .sum::<f64>()
        })
        .collect();
    let mut derivative = vec![0.0; x.nrows()];
    datafit.gradient(&eta, &mut derivative).unwrap();
    let input: Vec<_> = fit
        .coefficients()
        .iter()
        .enumerate()
        .map(|(j, &b)| {
            b - fit.step_size()
                * (0..x.nrows())
                    .map(|i| x[(i, j)] * derivative[i])
                    .sum::<f64>()
        })
        .collect();
    let mut output = vec![0.0; x.ncols()];
    penalty.prox(&input, fit.step_size(), &mut output).unwrap();
    let mapping = fit
        .coefficients()
        .iter()
        .zip(output)
        .map(|(b, z)| (b - z).abs() / fit.step_size())
        .fold(derivative.iter().sum::<f64>().abs(), f64::max);
    assert!(mapping <= TOLERANCE * 1.001, "original mapping {mapping}");
    let objective = datafit.value(&eta).unwrap() + penalty.value(fit.coefficients()).unwrap();
    assert!((objective - fit.objective()).abs() <= 1e-11 * (1.0 + objective.abs()));
}

fn measure<D, P, F, T>(
    label: &str,
    x: &Mat<f64>,
    datafit: &D,
    penalty: &P,
    repeats: usize,
    solve: F,
) where
    D: SmoothDatafit,
    P: ProximalPenalty,
    F: Fn(bool) -> T,
    T: FitSummary,
{
    let reference = solve(false);
    validate(x, datafit, penalty, &reference);
    let objective = reference.objective();
    drop(reference);
    let warm = solve(true);
    validate(x, datafit, penalty, &warm);
    assert!((warm.objective() - objective).abs() <= 1e-9 * (1.0 + objective.abs()));
    drop(warm);
    for enabled in [false, true] {
        let mut durations = Vec::with_capacity(repeats);
        for _ in 0..repeats {
            let start = Instant::now();
            let fit = black_box(solve(enabled));
            durations.push(start.elapsed());
            black_box(&fit);
        }
        durations.sort_unstable();
        let seconds = durations[repeats / 2].as_secs_f64();
        let profiler = dhat::Profiler::builder().testing().build();
        let fit = solve(enabled);
        let stats = dhat::HeapStats::get();
        let diagnostics = fit.screening_diagnostics().clone();
        let iterations = fit.iterations();
        let mapping = fit.stopping_value();
        let fit_objective = fit.objective();
        drop(fit);
        drop(profiler);
        println!(
            "{label},{enabled},{seconds:.9},{iterations},{mapping:.3e},{fit_objective:.12e},{},{},{},{},{},{},{},{}",
            diagnostics.certificate_passes,
            diagnostics.zero_tests,
            diagnostics.screened_features,
            diagnostics.active_dimension,
            diagnostics.rebuilds,
            stats.total_blocks,
            stats.total_bytes,
            stats.max_bytes
        );
    }
}

fn main() {
    let repeats = if std::env::args().any(|arg| arg == "--quick" || arg == "--test") {
        1
    } else {
        5
    };
    let (n, p) = (160, 48);
    println!(
        "loss,regime,penalty,solver,n,p,screening,median_seconds,iterations,original_mapping,objective,certificate_passes,zero_tests,screened_features,active_dimension,rebuilds,allocations,allocated_bytes,peak_live_bytes"
    );
    for sparse_solution in [true, false] {
        let regime = if sparse_solution { "sparse" } else { "weak" };
        let (dense, csc, gaussian, binomial) = dataset(n, p, sparse_solution);
        macro_rules! run {
            ($loss_name:literal, $loss:expr, $strength:expr, $name:literal, $penalty:expr) => {{
                let datafit = $loss;
                let strength: f64 = $strength;
                let penalty = ($penalty)(strength);
                let problem =
                    Problem::new(&csc, datafit, penalty.clone()).normalize(Normalization::None);
                let pg = ProximalGradient::<Col<f64>>::new()
                    .tolerance(TOLERANCE)
                    .max_iterations(100_000);
                let hybrid = Hybrid::<Col<f64>>::new()
                    .tolerance(TOLERANCE)
                    .max_iterations(100_000);
                let label = format!("{},{regime},{},pg,{n},{p}", $loss_name, $name);
                measure(&label, &dense, &datafit, &penalty, repeats, |enabled| {
                    problem.fit_with(&pg.clone().screening(enabled)).unwrap()
                });
                let label = format!("{},{regime},{},hybrid,{n},{p}", $loss_name, $name);
                measure(&label, &dense, &datafit, &penalty, repeats, |enabled| {
                    problem
                        .fit_with(&hybrid.clone().screening(enabled))
                        .unwrap()
                });
            }};
        }
        macro_rules! penalties {
            ($loss_name:literal, $loss:expr, $strength:expr) => {
                run!($loss_name, $loss, $strength, "clustered", |a| {
                    ClusteredLasso::new(a, a / (8.0 * p as f64))
                });
                run!($loss_name, $loss, $strength, "chain", |a| {
                    ChainFusedLasso::new(a, a / 8.0)
                });
                run!($loss_name, $loss, $strength, "slope", |a| Slope::new(
                    (0..p)
                        .map(|j| a * (1.0 + 0.2 * (p - j) as f64 / p as f64))
                        .collect()
                ));
            };
        }
        penalties!(
            "gaussian",
            Gaussian::new(&gaussian),
            if sparse_solution { 0.03 } else { 1e-6 }
        );
        penalties!(
            "binomial",
            Binomial::new(&binomial),
            if sparse_solution { 0.003 } else { 1e-6 }
        );
    }
}
