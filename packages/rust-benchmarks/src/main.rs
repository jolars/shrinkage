//! Native command worker for Metewand's version-1 fresh-fit lifecycle.

mod elastic_net;

use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;

use serde::Deserialize;
use serde_json::{Value, json};

use elastic_net::{Dataset, Penalty, Prepared, Role, evaluate};

type WorkerResult<T> = Result<T, Box<dyn Error>>;

#[derive(Deserialize)]
struct Request {
    id: String,
    #[serde(flatten)]
    method: Method,
}

#[derive(Deserialize)]
#[serde(tag = "method", rename_all = "snake_case")]
enum Method {
    Hello {
        protocols: Vec<u32>,
        role: String,
        worker_id: String,
    },
    Prepare {
        dataset_dir: PathBuf,
        problem_parameters: Penalty,
    },
    Execute {
        result_dir: PathBuf,
        observation_control: Value,
    },
    Evaluate {
        dataset_dir: PathBuf,
        problem_parameters: Penalty,
        problem_semantics: Semantics,
        result_dir: PathBuf,
        metrics_path: PathBuf,
    },
    Reset,
    Shutdown,
}

#[derive(Deserialize)]
struct Semantics {
    one_shot_completion: Completion,
}

#[derive(Deserialize)]
struct Completion {
    kkt_violation: f64,
}

#[derive(Deserialize)]
struct ResultEnvelope {
    data: Coefficients,
}

#[derive(Deserialize)]
struct Coefficients {
    coefficients: Vec<f64>,
}

struct Worker {
    role: Role,
    prepared: Option<Prepared>,
}

impl Worker {
    fn respond(&mut self, id: &str, method: Method) -> WorkerResult<Value> {
        match method {
            Method::Hello {
                protocols,
                role,
                worker_id,
            } => {
                let (expected_role, capabilities) = match self.role {
                    Role::Evaluator => ("problem_evaluator", &["problem_semantics"][..]),
                    _ => ("implementation", &["fresh_sequence"][..]),
                };
                if !protocols.contains(&1) || role != expected_role {
                    return Err("expected protocol 1 and the worker's declared role".into());
                }
                Ok(json!({
                    "id": id, "ok": true, "protocol": 1, "worker_id": worker_id,
                    "capabilities": capabilities, "sdk": null,
                }))
            }
            Method::Prepare {
                dataset_dir,
                problem_parameters,
            } => {
                self.prepared = None;
                let data = Dataset::read(&dataset_dir)?;
                self.prepared = Some(Prepared::new(self.role, data, problem_parameters)?);
                Ok(json!({"id": id, "ok": true}))
            }
            Method::Execute {
                result_dir,
                observation_control,
            } => {
                let prepared = self
                    .prepared
                    .take()
                    .ok_or("execute requires a prepared fit")?;
                let tolerance = observation_control["tolerance"]
                    .as_f64()
                    .filter(|tolerance| *tolerance > 0.0)
                    .ok_or("execute requires a positive tolerance")?;
                let model = prepared.fit(tolerance)?;
                let result = json!({
                    "schema_version": 1, "data": {"coefficients": model.coefficients}, "files": [],
                });
                fs::write(result_dir.join("result.json"), serde_json::to_vec(&result)?)?;
                Ok(json!({
                    "id": id, "ok": true, "result": {"manifest": "result.json"},
                    "implementation_time_ns": model.implementation_time_ns,
                    "observation_control": observation_control, "statistics": model.statistics,
                }))
            }
            Method::Evaluate {
                dataset_dir,
                problem_parameters,
                problem_semantics,
                result_dir,
                metrics_path,
            } => {
                if !matches!(self.role, Role::Evaluator) {
                    return Err("evaluate requires the evaluator role".into());
                }
                let data = Dataset::read(&dataset_dir)?;
                let result: ResultEnvelope =
                    serde_json::from_slice(&fs::read(result_dir.join("result.json"))?)?;
                let (metrics, verdict) =
                    match evaluate(&data, &result.data.coefficients, problem_parameters) {
                        Some(metrics) => {
                            let target = problem_semantics.one_shot_completion.kkt_violation;
                            let verdict = if metrics.kkt_violation <= target {
                                "reached"
                            } else {
                                "not_reached"
                            };
                            (serde_json::to_value(metrics)?, verdict)
                        }
                        None => (json!({}), "invalid"),
                    };
                let output = json!({"schema_version": 1, "data": metrics});
                fs::write(metrics_path, serde_json::to_vec(&output)?)?;
                Ok(json!({"id": id, "ok": true, "verdict": verdict}))
            }
            Method::Reset | Method::Shutdown => {
                self.prepared = None;
                Ok(json!({"id": id, "ok": true}))
            }
        }
    }
}

fn serve(role: Role) -> WorkerResult<()> {
    let read_fd = std::env::var("METEWAND_PROTOCOL_READ_FD")?.parse::<u32>()?;
    let write_fd = std::env::var("METEWAND_PROTOCOL_WRITE_FD")?.parse::<u32>()?;
    // Open the inherited Unix pipes without taking ownership of the original descriptors.
    let input = BufReader::new(File::open(format!("/dev/fd/{read_fd}"))?);
    let output = OpenOptions::new()
        .write(true)
        .open(format!("/dev/fd/{write_fd}"))?;
    let mut output = BufWriter::new(output);
    let mut worker = Worker {
        role,
        prepared: None,
    };
    for line in input.lines() {
        let Request { id, method } = serde_json::from_str(&line?)?;
        let shutdown = matches!(method, Method::Shutdown);
        let response = worker.respond(&id, method).unwrap_or_else(|error| {
            json!({"id": id, "ok": false, "error": {"code": "operation_failed", "message": error.to_string()}})
        });
        serde_json::to_writer(&mut output, &response)?;
        output.write_all(b"\n")?;
        output.flush()?;
        if shutdown {
            break;
        }
    }
    Ok(())
}

fn main() -> WorkerResult<()> {
    let arguments: Vec<_> = std::env::args().skip(1).collect();
    match arguments.as_slice() {
        [role] if role == "shrinkage" => serve(Role::Shrinkage),
        [role] if role == "linfa" => serve(Role::Linfa),
        [role] if role == "evaluator" => serve(Role::Evaluator),
        [command, path] if command == "generate" => Dataset::generate(path.as_ref()),
        _ => Err("expected shrinkage, linfa, evaluator, or generate PATH".into()),
    }
}
