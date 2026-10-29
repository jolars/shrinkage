//! Complete fits and paths, including independently checked original-loss accuracy.
//!
//! Run with `--quick` for one timed repetition, or `--lasso-only` to restrict
//! evaluation to coordinate fits and paths. Heap profiling runs separately.

use std::{hint::black_box, time::Instant};

use faer::{
    Col, Mat,
    sparse::{SparseColMat, Triplet},
};
use shrinkage::{
    Binomial, ChainFusedLasso, ClusteredLasso, CoordinateDescent, Gaussian, Hybrid, HybridFit,
    Irls, IrlsFit, L1, LassoFit, Normalization, Problem, ProximalPenalty, Slope, SmoothDatafit,
    Termination, WorkingSetDiagnostics, WorkingSetPolicy,
};

#[global_allocator]
static ALLOCATOR: dhat::Alloc = dhat::Alloc;

const TOLERANCE: f64 = 1e-8;

trait Summary {
    fn coefficients(&self) -> &[f64];
    fn intercept(&self) -> f64;
    fn objective(&self) -> f64;
    fn termination(&self) -> Termination;
    fn iterations(&self) -> usize;
    fn working_set(&self) -> &WorkingSetDiagnostics;
    fn step(&self) -> f64 {
        1.0
    }
    fn certificates(&self) -> usize {
        0
    }
}

macro_rules! summary {
    ($fit:ty, $($extra:item),* $(,)?) => {
        impl Summary for $fit {
            fn coefficients(&self) -> &[f64] { self.coefficients() }
            fn intercept(&self) -> f64 { self.intercept() }
            fn objective(&self) -> f64 { self.objective() }
            fn termination(&self) -> Termination { self.termination() }
            fn iterations(&self) -> usize { self.iterations() }
            fn working_set(&self) -> &WorkingSetDiagnostics { self.working_set_diagnostics() }
            $($extra)*
        }
    };
}

summary!(
    LassoFit,
    fn certificates(&self) -> usize {
        self.working_set_diagnostics().full_checks
    }
);
summary!(IrlsFit,);
summary!(
    HybridFit,
    fn step(&self) -> f64 {
        self.step_size()
    },
    fn certificates(&self) -> usize {
        self.screening_diagnostics().certificate_passes
    }
);

fn dataset(
    n: usize,
    p: usize,
    sparse: bool,
) -> (Mat<f64>, SparseColMat<usize, f64>, Vec<f64>, Vec<f64>) {
    let mut seed = 20261029_u64;
    let x = Mat::from_fn(n, p, |_, _| {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let value = (seed >> 11) as f64 / (1_u64 << 53) as f64;
        if value < 0.15 {
            4.0 * (value / 0.15 - 0.5)
        } else {
            0.0
        }
    });
    let triplets: Vec<_> = (0..p)
        .flat_map(|j| {
            let x = &x;
            (0..n).filter_map(move |i| (x[(i, j)] != 0.0).then_some(Triplet::new(i, j, x[(i, j)])))
        })
        .collect();
    let csc = SparseColMat::try_new_from_triplets(n, p, &triplets).unwrap();
    let gaussian: Vec<_> = (0..n)
        .map(|i| {
            0.3 + (0..p)
                .map(|j| {
                    let b = if sparse {
                        match j {
                            0 => 1.2,
                            1 => -0.8,
                            2 => 0.5,
                            _ => 0.0,
                        }
                    } else {
                        0.08 * ((j % 7) as f64 - 3.0)
                    };
                    x[(i, j)] * b
                })
                .sum::<f64>()
        })
        .collect();
    let binomial = gaussian
        .iter()
        .map(|eta| 1.0 / (1.0 + (-eta).exp()))
        .collect();
    (x, csc, gaussian, binomial)
}

fn validate<D: SmoothDatafit, P: ProximalPenalty, F: Summary>(
    x: &Mat<f64>,
    datafit: &D,
    penalty: &P,
    fit: &F,
) -> f64 {
    assert_eq!(fit.termination(), Termination::Converged);
    let eta: Vec<_> = (0..x.nrows())
        .map(|i| {
            fit.intercept()
                + fit
                    .coefficients()
                    .iter()
                    .enumerate()
                    .map(|(j, b)| x[(i, j)] * b)
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
            b - fit.step()
                * (0..x.nrows())
                    .map(|i| x[(i, j)] * derivative[i])
                    .sum::<f64>()
        })
        .collect();
    let mut output = vec![0.0; x.ncols()];
    penalty.prox(&input, fit.step(), &mut output).unwrap();
    let mapping = fit
        .coefficients()
        .iter()
        .zip(output)
        .map(|(b, z)| (b - z).abs() / fit.step())
        .fold(derivative.iter().sum::<f64>().abs(), f64::max);
    assert!(mapping <= 1.01 * TOLERANCE, "original mapping {mapping}");
    let objective = datafit.value(&eta).unwrap() + penalty.value(fit.coefficients()).unwrap();
    assert!((objective - fit.objective()).abs() <= 1e-10 * (1.0 + objective.abs()));
    mapping
}

fn policies() -> Vec<(&'static str, WorkingSetPolicy)> {
    let mut policies = vec![
        ("disabled", WorkingSetPolicy::Disabled),
        ("ever_active", WorkingSetPolicy::EverActive),
    ];
    for initial_size in [10, 100, 200] {
        for prune in [false, true] {
            policies.push((
                if prune {
                    "adaptive_pruned"
                } else {
                    "adaptive_monotone"
                },
                WorkingSetPolicy::Adaptive {
                    initial_size,
                    prune,
                },
            ));
        }
    }
    policies
}

fn measure<D, P, F>(
    label: &str,
    x: &Mat<f64>,
    datafit: &D,
    penalties: &[P],
    repeats: usize,
    supports_screening: bool,
    solve: impl Fn(WorkingSetPolicy, bool) -> Vec<F>,
) where
    D: SmoothDatafit,
    P: ProximalPenalty,
    F: Summary,
{
    let reference = solve(WorkingSetPolicy::Disabled, false);
    for (fit, penalty) in reference.iter().zip(penalties) {
        validate(x, datafit, penalty, fit);
    }
    for (name, policy) in policies() {
        for &screening in if supports_screening {
            &[false, true][..]
        } else {
            &[false][..]
        } {
            let warm = solve(policy, screening);
            for ((fit, penalty), reference) in warm.iter().zip(penalties).zip(&reference) {
                validate(x, datafit, penalty, fit);
                assert!(
                    (fit.objective() - reference.objective()).abs()
                        < 1e-8 * (1.0 + reference.objective().abs()),
                    "{label},{name},{policy:?},screening={screening}: objective {} differs from reference {}",
                    fit.objective(),
                    reference.objective()
                );
            }
            drop(warm);
            let mut durations = Vec::with_capacity(repeats);
            for _ in 0..repeats {
                let start = Instant::now();
                let fits = black_box(solve(policy, screening));
                durations.push(start.elapsed());
                black_box(&fits);
            }
            durations.sort_unstable();
            let profiler = dhat::Profiler::builder().testing().build();
            let fits = solve(policy, screening);
            let heap = dhat::HeapStats::get();
            let (
                mut iterations,
                mut full,
                mut restricted,
                mut rebuilds,
                mut expansions,
                mut reactivations,
                mut features,
                mut blocks,
                mut peak,
                mut certificates,
            ) = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
            let active = fits.last().unwrap().working_set().active_dimension;
            for fit in &fits {
                let work = fit.working_set();
                iterations += fit.iterations();
                full += work.full_checks;
                restricted += work.restricted_checks;
                rebuilds += work.rebuilds;
                expansions += work.expansions;
                reactivations += work.reactivations;
                features += work.feature_visits;
                blocks += work.block_visits;
                peak = peak.max(work.peak_active_dimension);
                certificates += fit.certificates();
            }
            // Validation allocates independent reference storage and therefore
            // belongs outside both timing and the heap measurement.
            drop(profiler);
            let mapping = fits
                .iter()
                .zip(penalties)
                .map(|(fit, penalty)| validate(x, datafit, penalty, fit))
                .fold(0.0, f64::max);
            let initial = if let WorkingSetPolicy::Adaptive { initial_size, .. } = policy {
                initial_size
            } else {
                0
            };
            println!(
                "{label},{name},{initial},{screening},{:.9},{iterations},{mapping:.3e},{full},{restricted},{certificates},{rebuilds},{expansions},{reactivations},{active},{peak},{features},{blocks},{},{},{}",
                durations[repeats / 2].as_secs_f64(),
                heap.total_blocks,
                heap.total_bytes,
                heap.max_bytes
            );
        }
    }
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let repeats = if args.iter().any(|arg| arg == "--quick" || arg == "--test") {
        1
    } else {
        5
    };
    let lasso_only = args.iter().any(|arg| arg == "--lasso-only");
    let (n, p) = (160, 256);
    println!(
        "storage,regime,loss,solver,mode,n,p,policy,initial_size,screening,median_seconds,iterations,original_mapping,full_checks,restricted_checks,certificate_passes,rebuilds,expansions,reactivations,active_dimension,peak_dimension,feature_visits,block_visits,allocations,allocated_bytes,peak_live_bytes"
    );
    for sparse in [true, false] {
        let (dense, csc, gaussian, binomial) = dataset(n, p, sparse);
        let regime = if sparse { "sparse" } else { "weak" };
        let strength = if sparse { 0.03 } else { 0.001 };
        let strengths = [4.0, 2.0, 1.0, 0.5].map(|factor| L1::new(strength * factor));
        macro_rules! storage {
            ($storage:literal, $x:expr) => {{
                let x = $x;
                let datafit = Gaussian::new(&gaussian);
                let problem = Problem::new(x, datafit, L1::new(strength)).normalize(Normalization::None);
                for path in [false, true] {
                    let mode = if path { "path" } else { "single" };
                    let penalties = if path { strengths.to_vec() } else { vec![L1::new(strength)] };
                    let label = format!("{},{regime},gaussian,coordinate,{mode},{n},{p}", $storage);
                    measure(&label, &dense, &datafit, &penalties, repeats, path, |policy, screening| {
                        let solver = CoordinateDescent::new().tolerance(TOLERANCE)
                            .working_set_policy(policy).screening(screening).max_iterations(100_000);
                        if path { problem.fit_path_with(&solver, penalties.clone()).unwrap().into_parts().0 }
                        else { vec![problem.fit_with(&solver).unwrap()] }
                    });
                }
                if !lasso_only {
                    let datafit = Binomial::new(&binomial);
                    let problem = Problem::new(x, datafit, L1::new(strength / 4.0)).normalize(Normalization::None);
                    for path in [false, true] {
                        let mode = if path { "path" } else { "single" };
                        let penalties: Vec<_> = if path { strengths.iter().map(|a| L1::new(a.lambda() / 4.0)).collect() }
                            else { vec![L1::new(strength / 4.0)] };
                        let label = format!("{},{regime},binomial,irls,{mode},{n},{p}", $storage);
                        measure(&label, &dense, &datafit, &penalties, repeats, false, |policy, _| {
                            let solver = Irls::new().tolerance(TOLERANCE).working_set_policy(policy).max_iterations(1_000);
                            if path { problem.fit_path_with(&solver, penalties.clone()).unwrap().into_parts().0 }
                            else { vec![problem.fit_with(&solver).unwrap()] }
                        });
                    }
                    macro_rules! hybrid {
                        ($loss_name:literal, $datafit:expr, $name:literal, $penalty:expr) => {{
                            let datafit = $datafit;
                            let penalty = $penalty;
                            let problem = Problem::new(x, datafit, penalty.clone()).normalize(Normalization::None);
                            let label = format!("{},{regime},{},{},single,{n},{p}", $storage, $loss_name, $name);
                            measure(&label, &dense, &datafit, &[penalty], repeats, true, |policy, screening| {
                                vec![problem.fit_with(&Hybrid::<Col<f64>>::new().tolerance(TOLERANCE)
                                    .working_set_policy(policy).screening(screening).max_iterations(100_000)).unwrap()]
                            });
                        }};
                    }
                    macro_rules! penalties {
                        ($loss:literal, $datafit:expr, $a:expr) => {{
                            let a = $a;
                            hybrid!($loss, $datafit, "hybrid_clustered", ClusteredLasso::new(a, a / (8.0 * p as f64)));
                            hybrid!($loss, $datafit, "hybrid_chain", ChainFusedLasso::new(a, a / 8.0));
                            hybrid!($loss, $datafit, "hybrid_slope", Slope::new((0..p).map(|j| a * (1.0 + 0.2 * (p - j) as f64 / p as f64)).collect()));
                        }};
                    }
                    penalties!("gaussian", Gaussian::new(&gaussian), strength);
                    penalties!("binomial", Binomial::new(&binomial), strength / 4.0);
                }
            }};
        }
        storage!("dense", &dense);
        storage!("csc", &csc);
    }
}
