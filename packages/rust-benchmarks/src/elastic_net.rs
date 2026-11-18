//! Shared input, public API adapters, and independent elastic net metrics.

use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use linfa::traits::Fit;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use shrinkage::{ElasticNet, Gaussian, Irls, Normalization, Problem, Termination};
use smartcore::linear::elastic_net::{ElasticNet as SmartCoreElasticNet, ElasticNetParameters};

use crate::WorkerResult;

const N_SAMPLES: usize = 100;
const N_FEATURES: usize = 10_000;
const N_SIGNALS: usize = 10;
const FEATURES_PER_GROUP: usize = 8;
const DATASET_SEED: u64 = 20261026;
const LINFA_MAX_SWEEPS: u32 = 1_000;

#[derive(Deserialize, Serialize)]
pub(crate) struct Dataset {
    columns: Vec<Vec<f64>>,
    y: Vec<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    signals: Vec<(usize, f64)>,
}

impl Dataset {
    pub(crate) fn generate(path: &Path) -> WorkerResult<()> {
        let mut state = DATASET_SEED;
        let mut uniform = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            2.0 * (state >> 11) as f64 / (1_u64 << 53) as f64 - 1.0
        };

        let mut columns = Vec::with_capacity(N_FEATURES);
        for _ in 0..N_FEATURES / FEATURES_PER_GROUP {
            let shared: Vec<_> = (0..N_SAMPLES).map(|_| uniform()).collect();
            for _ in 0..FEATURES_PER_GROUP {
                columns.push(
                    shared
                        .iter()
                        .map(|value| value + 0.5 * uniform())
                        .collect::<Vec<_>>(),
                );
            }
        }
        let signals: Vec<_> = (0..N_SIGNALS)
            .map(|j| (j * (N_FEATURES / N_SIGNALS), [1.0, -1.0][j % 2]))
            .collect();
        let mut y: Vec<_> = (0..N_SAMPLES)
            .map(|i| {
                let signal: f64 = signals.iter().map(|&(j, beta)| columns[j][i] * beta).sum();
                signal + 0.05 * uniform()
            })
            .collect();
        let mean = y.iter().sum::<f64>() / N_SAMPLES as f64;
        y.iter_mut().for_each(|value| *value -= mean);
        // Keep SmartCore's response centering and mean intercept exactly zero.
        let (last, preceding) = y.split_last_mut().expect("nonempty benchmark response");
        *last = -preceding.iter().sum::<f64>();

        let data = Self {
            columns,
            y,
            seed: Some(DATASET_SEED),
            signals,
        };
        let mut writer = BufWriter::new(File::create(path)?);
        serde_json::to_writer(&mut writer, &data)?;
        writer.flush()?;
        Ok(())
    }

    pub(crate) fn read(directory: &Path) -> WorkerResult<Self> {
        let reader = BufReader::new(File::open(directory.join("data.json"))?);
        let data: Self = serde_json::from_reader(reader)?;
        if data.y.is_empty()
            || data.columns.is_empty()
            || data
                .columns
                .iter()
                .any(|column| column.len() != data.y.len())
            || data
                .y
                .iter()
                .chain(data.columns.iter().flatten())
                .any(|value| !value.is_finite())
        {
            return Err("expected a finite, rectangular design and a matching response".into());
        }
        Ok(data)
    }
}

#[derive(Clone, Copy, Deserialize)]
pub(crate) struct Penalty {
    lambda: f64,
    l1_ratio: f64,
}

#[derive(Clone, Copy)]
pub(crate) enum Role {
    Shrinkage,
    Linfa,
    SmartCore,
    Evaluator,
}

enum Design {
    Shrinkage(ndarray::Array2<f64>, Vec<f64>),
    Linfa(linfa::Dataset<f64, f64, ndarray_0_16::Ix1>),
    SmartCore(ndarray::Array2<f64>, Vec<f64>),
}

pub(crate) struct Prepared {
    design: Design,
    penalty: Penalty,
}

#[derive(Deserialize)]
#[serde(untagged)]
pub(crate) enum FitControl {
    Tolerance { tolerance: f64 },
    Iterations { iterations: usize },
}

pub(crate) struct Measurement {
    pub(crate) implementation_time_ns: u64,
    pub(crate) coefficients: Vec<f64>,
    pub(crate) statistics: Value,
}

impl Prepared {
    pub(crate) fn new(role: Role, data: Dataset, penalty: Penalty) -> WorkerResult<Self> {
        let shape = (data.y.len(), data.columns.len());
        let entries = data.columns.into_iter().flatten().collect();
        let design = match role {
            Role::Shrinkage => {
                use ndarray::ShapeBuilder;
                let x = ndarray::Array2::from_shape_vec(shape.f(), entries)?;
                Design::Shrinkage(x, data.y)
            }
            Role::Linfa => {
                use ndarray_0_16::ShapeBuilder;
                let x = ndarray_0_16::Array2::from_shape_vec(shape.f(), entries)?;
                let y = ndarray_0_16::Array1::from(data.y);
                Design::Linfa(linfa::DatasetBase::new(x, y))
            }
            Role::SmartCore => {
                use ndarray::ShapeBuilder;
                if data.y.iter().sum::<f64>() != 0.0 {
                    return Err("SmartCore requires the benchmark's zero-mean response".into());
                }
                let x = ndarray::Array2::from_shape_vec(shape.f(), entries)?;
                Design::SmartCore(x, data.y)
            }
            Role::Evaluator => return Err("the evaluator cannot prepare a fit".into()),
        };
        Ok(Self { design, penalty })
    }

    pub(crate) fn fit(self, control: FitControl) -> WorkerResult<Measurement> {
        let Penalty { lambda, l1_ratio } = self.penalty;
        match (self.design, control) {
            (Design::Shrinkage(x, y), FitControl::Tolerance { tolerance }) if tolerance > 0.0 => {
                let start = Instant::now();
                let fit = Problem::new(
                    &x,
                    Gaussian::new(&y).fit_intercept(false),
                    ElasticNet::from_alpha(lambda, l1_ratio),
                )
                .normalize(Normalization::None)
                .fit_with(&Irls::new().tolerance(tolerance))?;
                let implementation_time_ns = start.elapsed().as_nanos().try_into()?;
                let termination = match fit.termination() {
                    Termination::Converged => "converged",
                    Termination::IterationLimit => "iteration_limit",
                };
                Ok(Measurement {
                    implementation_time_ns,
                    coefficients: fit.coefficients().to_vec(),
                    statistics: json!({
                        "outer_iterations": fit.iterations(),
                        "coordinate_sweeps": fit.coordinate_sweeps(),
                        "termination": termination,
                    }),
                })
            }
            (Design::Linfa(dataset), FitControl::Tolerance { tolerance }) if tolerance > 0.0 => {
                let start = Instant::now();
                let fit = linfa_elasticnet::ElasticNet::params()
                    .penalty(lambda)
                    .l1_ratio(l1_ratio)
                    .with_intercept(false)
                    .max_iterations(LINFA_MAX_SWEEPS)
                    .tolerance(tolerance)
                    .fit(&dataset)?;
                let implementation_time_ns = start.elapsed().as_nanos().try_into()?;
                Ok(Measurement {
                    implementation_time_ns,
                    coefficients: fit.hyperplane().to_vec(),
                    statistics: json!({
                        "coordinate_sweeps": fit.n_steps(),
                        "iteration_limit_reached": fit.n_steps() == LINFA_MAX_SWEEPS,
                    }),
                })
            }
            (Design::SmartCore(x, y), FitControl::Iterations { iterations }) if iterations > 0 => {
                let start = Instant::now();
                // Match the shared objective after dividing SmartCore's objective by 2n.
                let scale = 1.0 + l1_ratio;
                let parameters = ElasticNetParameters::default()
                    .with_alpha(lambda * scale)
                    .with_l1_ratio(2.0 * l1_ratio / scale)
                    .with_normalize(false)
                    .with_lazy_normalization(true)
                    .with_tol(1e-9)
                    .with_max_iter(iterations);
                let fit = SmartCoreElasticNet::fit(&x, &y, parameters)?;
                let implementation_time_ns = start.elapsed().as_nanos().try_into()?;
                if *fit.intercept() != 0.0 {
                    return Err("SmartCore returned a nonzero intercept on centered data".into());
                }
                Ok(Measurement {
                    implementation_time_ns,
                    coefficients: fit.coefficients().column(0).to_vec(),
                    statistics: json!({}),
                })
            }
            _ => Err("expected a positive iteration cap for SmartCore or a positive tolerance for Shrinkage and Linfa".into()),
        }
    }
}

#[derive(Serialize)]
pub(crate) struct Metrics {
    pub(crate) objective: f64,
    pub(crate) duality_gap: f64,
    pub(crate) relative_gap: f64,
    pub(crate) kkt_violation: f64,
}

pub(crate) fn evaluate(data: &Dataset, beta: &[f64], penalty: Penalty) -> Option<Metrics> {
    if beta.len() != data.columns.len() || beta.iter().any(|value| !value.is_finite()) {
        return None;
    }

    let n = data.y.len() as f64;
    let null_model_loss = data.y.iter().map(|y| y * y).sum::<f64>() / (2.0 * n);
    if !null_model_loss.is_finite() || null_model_loss <= 0.0 {
        return None;
    }
    let l1 = penalty.lambda * penalty.l1_ratio;
    let l2 = penalty.lambda * (1.0 - penalty.l1_ratio);
    if !l1.is_finite() || l1 < 0.0 || !l2.is_finite() || l2 <= 0.0 {
        return None;
    }
    let sqrt_l2 = l2.sqrt();
    let residuals: Vec<_> = data
        .y
        .iter()
        .enumerate()
        .map(|(i, target)| {
            let prediction: f64 = data.columns.iter().zip(beta).map(|(x, b)| x[i] * b).sum();
            prediction - target
        })
        .collect();

    let loss = residuals.iter().map(|r| r * r).sum::<f64>() / (2.0 * n);
    let regularization: f64 = beta.iter().map(|b| l1 * b.abs() + 0.5 * l2 * b * b).sum();
    let objective = loss + regularization;
    if !objective.is_finite() {
        return None;
    }

    let mut duality_gap = 0.0;
    let mut kkt_violation = 0.0_f64;
    for (column, &b) in data.columns.iter().zip(beta) {
        let correlation = -column
            .iter()
            .zip(&residuals)
            .map(|(x, r)| x * r)
            .sum::<f64>()
            / n;
        let gradient = l2 * b - correlation;
        if !gradient.is_finite() {
            return None;
        }

        // The loss Fenchel gap vanishes for theta = (y - X beta) / n.
        // Splitting X^T theta into L1 and L2 terms gives a sum of nonnegative gaps.
        let l1_dual = correlation.clamp(-l1, l1);
        let l2_dual = correlation - l1_dual;
        let l1_gap = (l1 - b.signum() * l1_dual) * b.abs();
        let l2_gap = 0.5 * ((l2 * b - l2_dual) / sqrt_l2).powi(2);
        duality_gap += l1_gap + l2_gap;

        let violation = if b == 0.0 {
            (gradient.abs() - l1).max(0.0)
        } else {
            (gradient + l1 * b.signum()).abs()
        };
        kkt_violation = kkt_violation.max(violation);
    }
    let relative_gap = duality_gap / null_model_loss;
    if !relative_gap.is_finite() || !kkt_violation.is_finite() {
        return None;
    }
    Some(Metrics {
        objective,
        duality_gap,
        relative_gap,
        kkt_violation,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluator_scales_the_gap_by_the_null_model_loss() -> WorkerResult<()> {
        let data = Dataset {
            columns: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            y: vec![1.25, -1.25],
            seed: None,
            signals: vec![],
        };
        let penalty = Penalty {
            lambda: 0.125,
            l1_ratio: 0.5,
        };
        let metrics = evaluate(&data, &[1.25, -1.0], penalty).expect("valid coefficients");
        let metrics = serde_json::to_value(metrics)?;
        assert_eq!(metrics["duality_gap"], 0.126953125);
        assert_eq!(metrics["relative_gap"], 0.1625);
        Ok(())
    }

    #[test]
    fn smartcore_matches_the_shared_penalty_scale() -> WorkerResult<()> {
        let data = Dataset {
            columns: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            y: vec![1.25, -1.25],
            seed: None,
            signals: vec![],
        };
        let penalty = Penalty {
            lambda: 0.125,
            l1_ratio: 0.5,
        };
        let result = Prepared::new(Role::SmartCore, data, penalty)?
            .fit(FitControl::Iterations { iterations: 100 })?;
        assert_eq!(result.coefficients.len(), 2);
        assert!((result.coefficients[0] - 1.0).abs() < 1e-5);
        assert!((result.coefficients[1] + 1.0).abs() < 1e-5);
        Ok(())
    }
}
