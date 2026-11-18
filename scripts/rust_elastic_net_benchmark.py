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
METHODS = ("shrinkage", "linfa", "smartcore")
METEWAND_RUN_FAILED = 5
PROFILE_STOPS = {
    ("accepted", "target_reached"),
    ("censored", "schedule_exhausted"),
    ("censored", "active_time_limit"),
}


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
    tracked = {
        "shrinkage",
        "linfa",
        "linfa-elasticnet",
        "smartcore",
        "lazymatrix",
        "ndarray",
    }
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
            "smartcore_features": ["lazy-normalization", "ndarray-bindings", "std"],
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
    measurement = subprocess.run(
        [
            metewand,
            "run",
            "--manifest",
            directory / "metewand.yaml",
            "--output-dir",
            directory / "run",
        ],
        cwd=ROOT,
        check=False,
    )
    # Metewand reports budget-limited profiles with the same exit code as failures.
    if measurement.returncode != METEWAND_RUN_FAILED:
        measurement.check_returncode()
    run(
        metewand,
        "export",
        "--format",
        "json",
        "--output-dir",
        directory / "run",
        "--destination",
        directory / "analysis",
    )
    summary = summarize(directory)
    (directory / "elastic-net.png").write_bytes(render(summary))
    write_json(directory / "elastic-net.json", summary)
    return directory


def validate_export(snapshot):
    budget_limited = {
        profile["candidate_id"]
        for profile in snapshot["profiles"]
        if profile["status"] == "censored"
        and (profile["status"], profile["reason"]) in PROFILE_STOPS
    }
    censored_slots = {
        slot["id"]
        for slot in snapshot["slots"]
        if slot["state"] == "censored" and slot["candidate_id"] in budget_limited
    }
    failed_slots = {
        attempt["slot_id"]
        for attempt in snapshot["attempts"]
        if attempt["state"] == "failed"
    }
    for issue in snapshot["completeness"]["issues"]:
        if (
            issue["code"] == "unfinished_profile"
            and issue["candidate_id"] in budget_limited
        ) or (
            issue["code"] == "unsatisfied_slot"
            and issue["slot_id"] in censored_slots
            and issue["slot_id"] not in failed_slots
        ):
            continue
        raise ValueError(f"The Metewand run is incomplete. {issue['message']}")


def summarize(directory):
    snapshot = json.loads(
        (directory / "analysis/results.json").read_text(encoding="utf-8")
    )
    validate_export(snapshot)
    candidates = {
        candidate["implementation"]: candidate for candidate in snapshot["candidates"]
    }
    if len(snapshot["candidates"]) != len(candidates) or set(candidates) not in (
        {"shrinkage", "linfa"},
        set(METHODS),
    ):
        raise ValueError("Expected one configuration per benchmarked solver.")

    parameters = candidates["shrinkage"]["parameters"]["problem"]
    target = candidates["shrinkage"]["observation_policy"]["profile"]["policy"][
        "threshold"
    ]
    repetitions = candidates["shrinkage"]["measurement_repetitions"]
    build = json.loads((directory / "build.json").read_text(encoding="utf-8"))
    data_path = directory / "data.json"
    dataset = json.loads(data_path.read_text(encoding="utf-8"))
    y = dataset["y"]
    null_model_loss = math.fsum(value * value for value in y) / (2 * len(y))
    if not 0.0 < null_model_loss < math.inf:
        raise ValueError(
            "The relative duality gap requires a positive finite null-model loss."
        )
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
    profile_stops = {}
    for method in METHODS:
        if method not in candidates:
            continue
        candidate = candidates[method]
        policy = candidate["observation_policy"]["profile"]["policy"]
        if (
            candidate["parameters"]["problem"] != parameters
            or candidate["measurement_repetitions"] != repetitions
        ):
            raise ValueError("Expected a shared problem and repetition count.")
        if (
            policy.get("threshold") != target
            or policy.get("metric") != "relative_gap"
            or policy.get("direction") != "at_most"
        ):
            raise ValueError("Expected a shared relative duality gap target.")
        profiles = sorted(
            (
                profile
                for profile in snapshot["profiles"]
                if profile["candidate_id"] == candidate["id"]
            ),
            key=lambda profile: profile["measurement_index"],
        )
        if [profile["measurement_index"] for profile in profiles] != list(
            range(repetitions)
        ) or any(
            (profile["status"], profile["reason"]) not in PROFILE_STOPS
            for profile in profiles
        ):
            raise ValueError("Expected every repetition to reach its target or budget.")
        stops = [profile["reason"] for profile in profiles]
        profile_stops[method] = stops
        methods[method] = profile_points(
            [row for row in rows if row["implementation"] == method],
            target,
            stops,
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
            "n_samples": len(y),
            "n_features": len(dataset["columns"]),
            "null_model_loss": null_model_loss,
            "seed": dataset["seed"],
            "signals": dataset.get("signals", []),
            "artifact": artifacts.pop(),
            "data_sha256": digest(data_path),
        },
        "parameters": parameters,
        "relative_gap_target": target,
        "repetitions": repetitions,
        "profile_stops": profile_stops,
        "methods": methods,
    }


def profile_points(rows, target, stops):
    sequences = defaultdict(list)
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
                0.0 <= metrics.get(key, math.nan) < math.inf
                for key in ("objective", "duality_gap", "relative_gap", "kkt_violation")
            )
        ):
            raise ValueError(
                "Expected a valid fresh fit with finite metrics and a positive fit time."
            )
        sequences[row["measurement_index"]].append(row)

    if sorted(sequences) != list(range(len(stops))):
        raise ValueError("Expected measurements for every repetition.")
    groups = defaultdict(list)
    for repetition, stop in enumerate(stops):
        sequence = sorted(
            sequences[repetition], key=lambda row: row["profile"]["schedule_index"]
        )
        for index, row in enumerate(sequence):
            if row["profile"]["schedule_index"] != index:
                raise ValueError("Expected consecutive schedule points.")
            reached = row["metrics"]["relative_gap"] <= target
            if reached != (
                index == len(sequence) - 1 and stop == "target_reached"
            ) or row["completion"] != ("reached" if reached else "not_reached"):
                raise ValueError(
                    "Each profile must end at its first fit that meets the target."
                )
            groups[index].append(row)

    points = []
    for index in range(len(groups)):
        group = groups[index]
        control = group[0]["profile"]["observation_control"]
        if any(row["profile"]["observation_control"] != control for row in group):
            raise ValueError("Expected matching solver settings across repetitions.")
        samples = [
            {"time_ns": row["implementation_time_ns"], **row["metrics"]}
            for row in group
        ]
        points.append({**control, "measurements": samples})
    return points


def render(summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    null_model_loss = summary["dataset"]["null_model_loss"]
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11}):
        figure, ax = plt.subplots(figsize=(7.2, 4.6), layout="constrained")
        series = (
            ("shrinkage", "Shrinkage", "#2563a6", "o"),
            ("linfa", "Linfa", "#bb5725", "s"),
            ("smartcore", "SmartCore", "#267253", "^"),
        )
        for method, label, color, marker in series:
            if method not in summary["methods"]:
                continue
            times, gaps, lower_errors, upper_errors = [], [], [], []
            for point in summary["methods"][method]:
                measurements = point["measurements"]
                seconds = [row["time_ns"] / 1e9 for row in measurements]
                time = median(seconds)
                lower, _, upper = (
                    quantiles(seconds, n=4, method="inclusive")
                    if len(seconds) > 1
                    else (time, time, time)
                )
                times.append(time)
                lower_errors.append(time - lower)
                upper_errors.append(upper - time)
                gaps.append(
                    median(
                        row.get("relative_gap", row["duality_gap"] / null_model_loss)
                        for row in measurements
                    )
                )
            ax.plot(
                times,
                gaps,
                marker=marker,
                color=color,
                label=label,
                linewidth=1.8,
            )
            ax.errorbar(
                times,
                gaps,
                xerr=[lower_errors, upper_errors],
                fmt="none",
                ecolor=color,
                alpha=0.45,
                capsize=2,
            )

        ax.set_yscale("log")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Relative duality gap")
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
