"""Run the Metewand comparison or redraw its retained documentation figure."""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from statistics import median, quantiles

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "packages/python"
OUTPUT = PACKAGE / "docs/figures"
METHODS = ("shrinkage", "scikit-learn")


def candidate(snapshot, method):
    selected = [
        row for row in snapshot["candidates"] if row["implementation"] == method
    ]
    if len(selected) != 1:
        raise ValueError(f"Expected one configuration for {method}.")
    return selected[0]


def collect_observations(snapshot):
    if not snapshot["completeness"]["complete"]:
        raise ValueError("The Metewand comparison is incomplete.")
    rows = [
        row for row in snapshot["observations"] if row["selected"] and not row["warmup"]
    ]
    if {row["implementation"] for row in rows} != set(METHODS):
        raise ValueError("Expected measurements for both lasso implementations.")
    hashes = {
        snapshot["provenance"][row["provenance_id"]]["dataset"]["output"]["sha256"]
        for row in rows
    }
    if len(hashes) != 1:
        raise ValueError("All measurements must use the same dataset.")
    for row in rows:
        if row["timing_scope"] != "execute_only":
            raise ValueError("Expected execute_only measurements.")
        if row["profile"]["delivery"] != "fresh_sequence":
            raise ValueError("Expected independent fresh-sequence fits.")
        if row["implementation_time_ns"] is None or row["implementation_time_ns"] <= 0:
            raise ValueError("The worker did not report a positive fit time.")
        pools = row["statistics"]["threadpools"]
        if not pools or any(pool["num_threads"] != 1 for pool in pools):
            raise ValueError("Numerical libraries must use one thread.")
    return rows


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
            str(directory / "analysis"),
            "--format",
            "json",
            "--require-complete",
        ],
    )
    for command in commands:
        subprocess.run([executable, *command], cwd=ROOT, env=environment, check=True)
    source = directory / "analysis/results.json"
    snapshot = json.loads(source.read_text())
    for method in METHODS:
        convergence_points(snapshot, method)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, OUTPUT / "lasso.json")
    print(f"Retained Metewand run: {directory.relative_to(ROOT)}", flush=True)
    return snapshot


def convergence_points(snapshot, method):
    rows = [
        row for row in collect_observations(snapshot) if row["implementation"] == method
    ]
    repetitions = candidate(snapshot, method)["measurement_repetitions"]
    points = []

    def control(row):
        return row["profile"]["observation_control"]["iterations"]

    for limit in sorted({control(row) for row in rows}):
        selected = [row for row in rows if control(row) == limit]
        indices = {row["measurement_index"] for row in selected}
        if len(selected) != repetitions or indices != set(range(repetitions)):
            raise ValueError(
                "Every plotted control must have the declared repetitions."
            )
        if repetitions < 2:
            raise ValueError("Error bars require at least two declared repetitions.")
        times = [row["implementation_time_ns"] / 1e6 for row in selected]
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

    observations = collect_observations(snapshot)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11}):
        figure, ax = plt.subplots(figsize=(7.2, 4.6), layout="constrained")
        colors = ("#2563a6", "#bb5725")
        for method, color in zip(METHODS, colors, strict=True):
            points = convergence_points(snapshot, method)
            times = np.array([point["time_ms"] for point in points])
            gaps = np.array([point["duality_gap"] for point in points])
            version = next(
                row["statistics"]["version"]
                for row in observations
                if row["implementation"] == method
            )
            ax.plot(
                times,
                gaps,
                "o-",
                color=color,
                linewidth=2,
                markersize=4.5,
                label=f"{method} {version}",
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
        settings = candidate(snapshot, METHODS[0])
        parameters = settings["parameters"]["dataset"]
        repetitions = settings["measurement_repetitions"]
        ax.text(
            0,
            1.04,
            f"{parameters['n_samples']:,} samples · "
            f"{parameters['n_features']:,} features · "
            f"{repetitions} profiles · one thread",
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
