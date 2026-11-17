//! Shared input, public API adapters, and independent elastic net metrics.

use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use linfa::traits::Fit;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use shrinkage::{ElasticNet, Gaussian, Irls, Normalization, Problem, Termination};

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
        let y = (0..N_SAMPLES)
            .map(|i| {
                let signal: f64 = signals.iter().map(|&(j, beta)| columns[j][i] * beta).sum();
                signal + 0.05 * uniform()
            })
            .collect();

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
    Evaluator,
}

enum Design {
    Shrinkage(ndarray::Array2<f64>, Vec<f64>),
    Linfa(linfa::Dataset<f64, f64, ndarray_0_16::Ix1>),
}

pub(crate) struct Prepared {
    design: Design,
    penalty: Penalty,
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
            Role::Evaluator => return Err("the evaluator cannot prepare a fit".into()),
        };
        Ok(Self { design, penalty })
    }

    pub(crate) fn fit(self, tolerance: f64) -> WorkerResult<Measurement> {
        let Penalty { lambda, l1_ratio } = self.penalty;
        match self.design {
            Design::Shrinkage(x, y) => {
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
            Design::Linfa(dataset) => {
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
        }
    }
}

#[derive(Serialize)]
pub(crate) struct Metrics {
    pub(crate) objective: f64,
    pub(crate) kkt_violation: f64,
}

pub(crate) fn evaluate(data: &Dataset, beta: &[f64], penalty: Penalty) -> Option<Metrics> {
    if beta.len() != data.columns.len() || beta.iter().any(|value| !value.is_finite()) {
        return None;
    }

    let n = data.y.len() as f64;
    let l1 = penalty.lambda * penalty.l1_ratio;
    let l2 = penalty.lambda * (1.0 - penalty.l1_ratio);
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

    let mut kkt_violation = 0.0_f64;
    for (column, &b) in data.columns.iter().zip(beta) {
        let gradient = column
            .iter()
            .zip(&residuals)
            .map(|(x, r)| x * r)
            .sum::<f64>()
            / n
            + l2 * b;
        if !gradient.is_finite() {
            return None;
        }
        let violation = if b == 0.0 {
            (gradient.abs() - l1).max(0.0)
        } else {
            (gradient + l1 * b.signum()).abs()
        };
        kkt_violation = kkt_violation.max(violation);
    }
    Some(Metrics {
        objective,
        kkt_violation,
    })
}
