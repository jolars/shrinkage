# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib==3.10.7", "numpy==2.3.5"]
# ///
"""Run the native Metewand comparison or redraw the saved docs figure."""

import argparse
import json
import math
import os
import shutil
import subprocess
import tomllib
from collections import defaultdict
from datetime import UTC, datetime
from hashlib import file_digest, sha256
from io import BytesIO
from pathlib import Path
from statistics import median, quantiles
from tempfile import mkdtemp

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "packages/rust-benchmarks"
FIGURES = ROOT / "docs/rust/figures"
METHODS = ("shrinkage", "linfa")


def metewand_executable():
    sibling = ROOT.parent / "metewand/target/release/metewand"
    default = str(sibling) if sibling.is_file() else "metewand"
    executable = shutil.which(os.environ.get("METEWAND", default))
    if executable is None:
        raise SystemExit("Build ../metewand in release mode or set METEWAND.")
    return Path(executable).resolve()


def run(*command):
    subprocess.run(command, cwd=ROOT, check=True)


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def write_json(path, data):
    path.write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def source_digest():
    paths = [ROOT / "Cargo.toml", BENCHMARK / "Cargo.toml", BENCHMARK / "Cargo.lock"]
    paths.extend((ROOT / "src").rglob("*.rs"))
    paths.extend((BENCHMARK / "src").rglob("*.rs"))
    files = {path.relative_to(ROOT).as_posix(): digest(path) for path in sorted(paths)}
    return sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def build_worker(directory):
    target = ROOT / "target/rust-benchmarks"
    source_sha256 = source_digest()
    run(
        "cargo",
        "build",
        "--locked",
        "--release",
        "--manifest-path",
        BENCHMARK / "Cargo.toml",
        "--target-dir",
        target,
    )
    if source_digest() != source_sha256:
        raise RuntimeError("Worker sources changed while Cargo was building.")
    worker = directory / "worker"
    shutil.copy2(target / "release/shrinkage-rust-benchmarks", worker)
    shutil.copyfile(BENCHMARK / "Cargo.lock", directory / "Cargo.lock")
    lock = tomllib.loads((directory / "Cargo.lock").read_text(encoding="utf-8"))
    tracked = {"shrinkage", "linfa", "linfa-elasticnet", "lazymatrix", "ndarray"}
    versions = defaultdict(list)
    for package in lock["package"]:
        if package["name"] in tracked:
            versions[package["name"]].append(package["version"])
    write_json(
        directory / "build.json",
        {
            "profile": "release",
            "rustc": subprocess.check_output(
                ["rustc", "--version"], cwd=ROOT, text=True
            ).strip(),
            "shrinkage_features": ["ndarray_v0_17"],
            "versions": dict(versions),
            "source_sha256": source_sha256,
            "worker_sha256": digest(worker),
            "lockfile_sha256": digest(directory / "Cargo.lock"),
        },
    )
    run(worker, "generate", directory / "data.json")


def prepare_comparison(metewand):
    reports = ROOT / ".build/rust-elastic-net"
    reports.mkdir(parents=True, exist_ok=True)
    directory = Path(mkdtemp(prefix="run-", dir=reports))
    print(f"Metewand work directory: {directory.relative_to(ROOT)}", flush=True)
    build_worker(directory)
    manifest = directory / "metewand.yaml"
    shutil.copyfile(BENCHMARK / manifest.name, manifest)
    run(metewand, "check", "--manifest", manifest, "--workers")
    return directory


def run_comparison():
    metewand = metewand_executable()
    directory = prepare_comparison(metewand)
    run(
        metewand,
        "run",
        "--manifest",
        directory / "metewand.yaml",
        "--output-dir",
        directory / "run",
    )
    run(
        metewand,
        "export",
        "--format",
        "json",
        "--require-complete",
        "--output-dir",
        directory / "run",
        "--destination",
        directory / "analysis",
    )
    summary = summarize(directory)
    (directory / "elastic-net.png").write_bytes(render(summary))
    write_json(directory / "elastic-net.json", summary)
    return directory


def summarize(directory):
    snapshot = json.loads(
        (directory / "analysis/results.json").read_text(encoding="utf-8")
    )
    if not snapshot["completeness"]["complete"]:
        raise ValueError("The Metewand run is incomplete.")
    candidates = {
        candidate["implementation"]: candidate for candidate in snapshot["candidates"]
    }
    if len(snapshot["candidates"]) != 2 or set(candidates) != set(METHODS):
        raise ValueError(
            "Expected one Shrinkage configuration and one Linfa configuration."
        )

    parameters = candidates["shrinkage"]["parameters"]["problem"]
    target = candidates["shrinkage"]["observation_policy"]["profile"]["policy"][
        "threshold"
    ]
    repetitions = candidates["shrinkage"]["measurement_repetitions"]
    build = json.loads((directory / "build.json").read_text(encoding="utf-8"))
    data_path = directory / "data.json"
    dataset = json.loads(data_path.read_text(encoding="utf-8"))
    rows = [
        row for row in snapshot["observations"] if row["selected"] and not row["warmup"]
    ]
    provenances = [snapshot["provenance"][row["provenance_id"]] for row in rows]
    artifacts = {entry["dataset"]["artifact"] for entry in provenances}
    workers = {
        entry["implementation"]["executable"]["executable_sha256"]
        for entry in provenances
    }
    if len(artifacts) != 1 or workers != {build["worker_sha256"]}:
        raise ValueError(
            "Expected one dataset and the staged native worker for all fits."
        )

    methods = {}
    for method in METHODS:
        candidate = candidates[method]
        policy = candidate["observation_policy"]["profile"]["policy"]
        if (
            candidate["parameters"]["problem"] != parameters
            or candidate["measurement_repetitions"] != repetitions
            or policy["metric"] != "kkt_violation"
            or policy["threshold"] != target
        ):
            raise ValueError(
                "Expected a shared problem, accuracy target, and repetition count."
            )
        methods[method] = profile_points(
            [row for row in rows if row["implementation"] == method],
            target,
            repetitions,
        )

    provenance = provenances[0]
    started = min(int(attempt["started_at_ns"]) for attempt in snapshot["attempts"])
    return {
        "date_utc": datetime.fromtimestamp(started // 1_000_000_000, UTC)
        .date()
        .isoformat(),
        "host": provenance["host"],
        "metewand_version": provenance["tool_version"],
        "metewand_sha256": provenance["runner"]["executable_sha256"],
        "build": build,
        "dataset": {
            "n_samples": len(dataset["y"]),
            "n_features": len(dataset["columns"]),
            "seed": dataset["seed"],
            "signals": dataset.get("signals", []),
            "artifact": artifacts.pop(),
            "data_sha256": digest(data_path),
        },
        "parameters": parameters,
        "kkt_target": target,
        "repetitions": repetitions,
        "methods": methods,
    }


def profile_points(rows, target, repetitions):
    groups = defaultdict(list)
    for row in rows:
        profile = row["profile"]
        time_ns = row["implementation_time_ns"]
        metrics = row["metrics"]
        if (
            not row["valid"]
            or row["timing_scope"] != "execute_only"
            or row["primary_time"] != "implementation_time"
            or time_ns is None
            or time_ns <= 0
            or row["primary_time_ns"] != time_ns
            or profile["delivery"] != "fresh_sequence"
            or not all(
                math.isfinite(metrics[key]) for key in ("kkt_violation", "objective")
            )
        ):
            raise ValueError(
                "Expected a valid fresh fit with both metrics and a positive fit time."
            )
        groups[profile["schedule_index"]].append(row)

    if sorted(groups) != list(range(len(groups))) or len(groups) < 2:
        raise ValueError("Expected consecutive schedule points leading to the target.")
    points = []
    for index in sorted(groups):
        group = sorted(groups[index], key=lambda row: row["measurement_index"])
        tolerances = {
            row["profile"]["observation_control"]["tolerance"] for row in group
        }
        if len(tolerances) != 1 or [row["measurement_index"] for row in group] != list(
            range(repetitions)
        ):
            raise ValueError(
                "Each plotted tolerance must have all declared repetitions."
            )
        final = index == len(groups) - 1
        samples = []
        for row in group:
            if (row["metrics"]["kkt_violation"] <= target) != final:
                raise ValueError(
                    "Each profile must end at its first fit that meets the target."
                )
            samples.append({"time_ns": row["implementation_time_ns"], **row["metrics"]})
        points.append({"tolerance": tolerances.pop(), "measurements": samples})
    return points


def render(summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11}):
        figure, ax = plt.subplots(figsize=(7.2, 4.6), layout="constrained")
        versions = summary["build"]["versions"]
        series = (
            (
                "shrinkage",
                f"Shrinkage {versions['shrinkage'][0]} (IRLS)",
                "#2563a6",
                "o",
            ),
            ("linfa", f"Linfa {versions['linfa-elasticnet'][0]}", "#bb5725", "s"),
        )
        for method, label, color, marker in series:
            times, violations, lower_errors, upper_errors = [], [], [], []
            for point in summary["methods"][method]:
                measurements = point["measurements"]
                milliseconds = [row["time_ns"] / 1e6 for row in measurements]
                time = median(milliseconds)
                lower, _, upper = quantiles(milliseconds, n=4, method="inclusive")
                times.append(time)
                lower_errors.append(time - lower)
                upper_errors.append(upper - time)
                violations.append(
                    max(1e-16, median(row["kkt_violation"] for row in measurements))
                )
            ax.plot(
                times,
                violations,
                marker=marker,
                color=color,
                label=label,
                linewidth=1.8,
            )
            ax.errorbar(
                times,
                violations,
                xerr=[lower_errors, upper_errors],
                fmt="none",
                ecolor=color,
                alpha=0.45,
                capsize=2,
            )

        ax.axhline(
            summary["kkt_target"],
            color="#596473",
            linestyle="--",
            linewidth=1,
            label="Shared accuracy target",
        )
        ax.set_yscale("log")
        ax.set_xlim(left=0)
        ax.set_xlabel("Rust fit time (ms)")
        ax.set_ylabel("Maximum KKT violation")
        ax.set_title("Elastic net convergence", loc="left", weight="bold", pad=26)
        shape = summary["dataset"]
        signals = shape.get("signals", [])
        signal_label = f"{len(signals)} signals · " if signals else ""
        ax.text(
            0,
            1.03,
            f"{shape['n_samples']:,} observations · {shape['n_features']:,} features · "
            f"{signal_label}{summary['repetitions']} profiles per solver",
            transform=ax.transAxes,
            fontsize=10,
            color="#4b5563",
        )
        ax.grid(color="#e5e7eb")
        ax.set_axisbelow(True)
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        image = BytesIO()
        figure.savefig(image, format="png", dpi=180)
        plt.close(figure)
    return image.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_mutually_exclusive_group()
    commands.add_argument(
        "--run", action="store_true", help="Measure and save a local run."
    )
    commands.add_argument(
        "--record",
        type=Path,
        metavar="RUN_DIRECTORY",
        help="Refresh the docs with a completed local run.",
    )
    commands.add_argument(
        "--check", action="store_true", help="Check the saved docs figure."
    )
    args = parser.parse_args()
    if args.run:
        directory = run_comparison()
        print(f"Saved results and plot to {directory.relative_to(ROOT)}")
        return

    data_path = FIGURES / "elastic-net.json"
    image_path = FIGURES / "elastic-net.png"
    if args.record:
        data = summarize(args.record)
    else:
        data = json.loads(data_path.read_text(encoding="utf-8"))
    image = render(data)
    if args.check:
        if image_path.read_bytes() != image:
            raise ValueError(
                "Regenerate the Rust elastic net figure from its saved measurements."
            )
        print("Rust elastic net figure matches the saved measurements.")
        return
    if args.record:
        FIGURES.mkdir(parents=True, exist_ok=True)
        write_json(data_path, data)
    image_path.write_bytes(image)
    print(f"Wrote {image_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
