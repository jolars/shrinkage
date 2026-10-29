"""Run the Metewand comparison or redraw its retained documentation figure."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from statistics import median, quantiles

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "packages/python"
OUTPUT = PACKAGE / "docs/figures"
METHODS = ("shrinkage", "scikit-learn")
REPETITIONS = 5


def benchmark_metadata(rows):
    contexts = [
        row["provenance"]["data"]["context"]
        for row in rows
        if row["record_type"] == "observation" and not row["warmup"]
    ]
    dataset_hashes = {
        context["dataset_materialization"]["output"]["sha256"] for context in contexts
    }
    if len(dataset_hashes) != 1:
        raise ValueError("All measurements must use the same dataset.")
    context = contexts[0]
    return {
        "dataset_parameters": context["comparison"]["parameters"]["dataset"],
        "dataset_seed": context["dataset_materialization"]["derivation"]["seed"][
            "value"
        ],
        "dataset_tree_sha256": dataset_hashes.pop(),
        "execution_policy": context["execution_policy"],
        "profile": "fresh_sequence",
        "repetitions": REPETITIONS,
        "time_coordinate": "implementation_time_ns",
    }


def collect_observations(rows):
    observations = {}
    for row in rows:
        if row["record_type"] == "attempt" and row.get("failure_code"):
            raise ValueError(f"Failed attempt: {row['failure_code']}")
        if row["record_type"] != "observation" or row["warmup"]:
            continue
        if not row["valid"]:
            raise ValueError(f"Observation {row['id']} is invalid.")
        if row["compatibility"]["timing_scope"] != "execute_only":
            raise ValueError("Expected execute_only measurements.")
        statistics = row["provenance"]["data"]["implementation_statistics"]
        profile = row["profile"]
        if profile["delivery"] != "fresh_sequence":
            raise ValueError("Expected a fresh-sequence convergence profile.")
        observation = observations.setdefault(
            row["id"],
            {
                "id": row["id"],
                "attempt_id": row["attempt_id"],
                "implementation": statistics["implementation"],
                "version": statistics["version"],
                "iterations": statistics["iterations"],
                "iteration_limit": profile["observation_control"]["iterations"],
                "repetition": profile["measurement_index"],
                "threadpools": statistics["threadpools"],
                "fit_time_ns": row["implementation_time_ns"],
                "timed_wall_time_ns": row["timed_wall_time_ns"],
                "metrics": {},
            },
        )
        observation["metrics"][row["metric_name"].removeprefix("/")] = row[
            "metric_value"
        ]
    observations = list(observations.values())
    for method in METHODS:
        curves = {
            row["repetition"] for row in observations if row["implementation"] == method
        }
        if curves != set(range(REPETITIONS)):
            raise ValueError(
                f"Expected {REPETITIONS} curves for {method}, got {len(curves)}."
            )
        for repetition in curves:
            points = sorted(
                (
                    row
                    for row in observations
                    if row["implementation"] == method
                    and row["repetition"] == repetition
                ),
                key=lambda row: row["iteration_limit"],
            )
            if points[-1]["metrics"]["relative_gap"] > 1e-8:
                raise ValueError(f"{method} curve did not reach the accuracy target.")
            if [row["iteration_limit"] for row in points] != list(
                range(1, points[-1]["iteration_limit"] + 1)
            ):
                raise ValueError(f"{method} curve has missing or duplicate controls.")
    for row in observations:
        if row["fit_time_ns"] is None or row["fit_time_ns"] <= 0:
            raise ValueError("The worker did not report a positive fit time.")
        if any(pool["num_threads"] != 1 for pool in row["threadpools"]):
            raise ValueError("A numerical library used more than one thread.")
    return observations


def run_comparison(executable: str):
    directory = (
        ROOT / ".build/python-lasso" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    )
    directory.mkdir(parents=True)
    manifest = PACKAGE / "benchmarks.yaml"
    environment = os.environ.copy()
    environment["PATH"] = (
        str(Path(sys.executable).parent) + os.pathsep + environment["PATH"]
    )
    commands = (
        ["check", "--manifest", str(manifest), "--workers"],
        ["run", "--manifest", str(manifest), "--output-dir", str(directory / "run")],
        [
            "export",
            "--output-dir",
            str(directory / "run"),
            "--destination",
            str(directory / "tables"),
        ],
    )
    for command in commands:
        subprocess.run([executable, *command], cwd=ROOT, env=environment, check=True)
    rows = [
        json.loads(line)
        for line in (directory / "tables/results.jsonl").read_text().splitlines()
    ]
    observations = collect_observations(rows)
    cpu = platform.processor()
    if Path("/proc/cpuinfo").exists():
        cpu = next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            cpu,
        )
    from shrinkage import _shrinkage

    source_status = subprocess.check_output(
        [
            "git",
            "status",
            "--porcelain",
            "--",
            "src",
            "Cargo.toml",
            "Cargo.lock",
            "packages/python/src",
            "packages/python/rust",
            "packages/python/Cargo.toml",
            "packages/python/Cargo.lock",
        ],
        cwd=ROOT,
        text=True,
    ).strip()
    snapshot = {
        "schema_version": 2,
        "measured_at_utc": datetime.now(UTC).isoformat(),
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_changes": source_status,
        "native_extension_sha256": hashlib.sha256(
            Path(_shrinkage.__file__).read_bytes()
        ).hexdigest(),
        "metewand": {
            "version": subprocess.check_output(
                [executable, "--version"], text=True
            ).strip(),
            "executable_sha256": hashlib.sha256(
                Path(executable).read_bytes()
            ).hexdigest(),
        },
        "software": {
            "python": platform.python_version(),
            **{
                name: version(name)
                for name in (
                    "shrinkage",
                    "scikit-learn",
                    "numpy",
                    "scipy",
                    "metewand",
                    "matplotlib",
                )
            },
        },
        "hardware": {"cpu": cpu, "platform": platform.platform()},
        "run_directory": str(directory.relative_to(ROOT)),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "uv_lock_sha256": hashlib.sha256(
            (PACKAGE / "uv.lock").read_bytes()
        ).hexdigest(),
        "benchmark": benchmark_metadata(rows),
        "observations": observations,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "lasso.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    print(f"Retained Metewand run: {directory.relative_to(ROOT)}", flush=True)
    return snapshot


def convergence_points(snapshot, method):
    rows = [row for row in snapshot["observations"] if row["implementation"] == method]
    points = []
    for limit in sorted({row["iteration_limit"] for row in rows}):
        selected = [row for row in rows if row["iteration_limit"] == limit]
        if len(selected) != REPETITIONS:
            raise ValueError("Every plotted control must have five measurements.")
        times = [row["fit_time_ns"] / 1e6 for row in selected]
        lower, _, upper = quantiles(times, n=4, method="inclusive")
        points.append(
            {
                "iteration_limit": limit,
                "time_ms": median(times),
                "time_lower_ms": lower,
                "time_upper_ms": upper,
                "duality_gap": median(
                    row["metrics"]["duality_gap"] for row in selected
                ),
            }
        )
    return points


def render(snapshot):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11}):
        figure, ax = plt.subplots(figsize=(7.2, 4.6), layout="constrained")
        colors = ("#2563a6", "#bb5725")
        for method, color in zip(METHODS, colors, strict=True):
            points = convergence_points(snapshot, method)
            times = np.array([point["time_ms"] for point in points])
            gaps = np.array([point["duality_gap"] for point in points])
            ax.plot(
                times,
                gaps,
                "o-",
                color=color,
                linewidth=2,
                markersize=4.5,
                label=f"{method} {snapshot['software'][method]}",
            )
            errors = np.array(
                [
                    [point["time_ms"] - point["time_lower_ms"] for point in points],
                    [point["time_upper_ms"] - point["time_ms"] for point in points],
                ]
            )
            ax.errorbar(
                times,
                gaps,
                xerr=errors,
                fmt="none",
                ecolor=color,
                alpha=0.35,
                capsize=2,
                linewidth=1,
            )
        ax.set_xscale("linear")
        ax.set_yscale("log")
        ax.set_xlim(left=0, right=ax.get_xlim()[1] * 1.05)
        ax.set_xlabel("Python fit time (ms)")
        ax.set_ylabel("Duality gap")
        ax.set_title(
            "Lasso convergence at one regularization strength",
            loc="left",
            weight="bold",
            pad=30,
        )
        parameters = snapshot["benchmark"]["dataset_parameters"]
        ax.text(
            0,
            1.04,
            f"{parameters['n_samples']:,} samples · "
            f"{parameters['n_features']:,} features · five profiles · one thread",
            transform=ax.transAxes,
            fontsize=10,
            color="#4b5563",
        )
        ax.grid(color="#e5e7eb")
        ax.set_axisbelow(True)
        ax.legend(frameon=False)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=180)
        plt.close(figure)
    return buffer.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", action="store_true", help="Run a new comparison first."
    )
    parser.add_argument(
        "--check", action="store_true", help="Check the retained figure."
    )
    args = parser.parse_args()
    if args.run and args.check:
        parser.error("Choose --run or --check.")
    if args.run:
        executable = shutil.which(os.environ.get("METEWAND", "metewand"))
        if executable is None:
            parser.error("Install the Metewand CLI or set METEWAND to its executable.")
        snapshot = run_comparison(executable)
    else:
        snapshot = json.loads((OUTPUT / "lasso.json").read_text())
    figure = render(snapshot)
    destination = OUTPUT / "lasso.png"
    if args.check:
        if destination.read_bytes() != figure:
            raise ValueError("Regenerate the lasso figure from its retained results.")
        print("Lasso benchmark figure matches the retained results.")
    else:
        destination.write_bytes(figure)
        print(f"Wrote {destination.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
