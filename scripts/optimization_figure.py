# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib==3.10.7", "numpy==2.3.5"]
# ///
"""Generate the explorer's static fallback from its native Rust scene evaluator."""

import argparse
import json
import subprocess
from io import StringIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from penalty_regions import portable_svg

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/pages/figures/optimization-explorer.svg"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = subprocess.run(
        [
            "cargo",
            "run",
            "--locked",
            "--quiet",
            "--manifest-path",
            "docs/widget/Cargo.toml",
            "--bin",
            "fixture",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    fixture = json.loads(result.stdout)
    scene = fixture["scene"]
    n = scene["grid_size"]
    positions = scene["grid_low"] + (np.arange(n) + 0.5) * scene["grid_step"]
    radius = scene["radius"]
    plt.rcParams.update(
        {"svg.hashsalt": "shrinkage-explorer", "font.family": "DejaVu Sans"}
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.4, 3.5), layout="constrained")
    shades = [
        "#c3c3c3",
        "#c9c9c9",
        "#d0d0d0",
        "#d6d6d6",
        "#dcdcdc",
        "#e2e2e2",
        "#e9e9e9",
        "#efefef",
        "#f5f5f5",
    ]
    for ax, field, title in zip(
        axes,
        ["loss", "objective"],
        ["Datafit + penalty region", "Penalized objective"],
        strict=True,
    ):
        values = np.array(scene[field]).reshape(n, n)
        minimum = (
            scene["loss_minimum"]
            if field == "loss"
            else scene["reference"]["objective"]
        )
        levels = minimum + np.array(
            [0.003, 0.01, 0.03, 0.07, 0.15, 0.3, 0.55, 0.85]
        ) * (values.max() - minimum)
        ax.contourf(
            positions,
            positions,
            values,
            levels=np.concatenate(
                ([min(values.min(), minimum)], levels, [values.max()])
            ),
            colors=shades,
            zorder=0,
        )
        ax.contour(
            positions,
            positions,
            values,
            levels=levels,
            colors=["#ffffff"],
            alpha=0.9,
            linewidths=0.9,
            zorder=3,
        )
        if field == "loss" and scene["reference"]["converged"]:
            if scene["loss_at_optimum"] > scene["loss_minimum"]:
                ax.contour(
                    positions,
                    positions,
                    values,
                    levels=[scene["loss_at_optimum"]],
                    colors=["#ffffff"],
                    alpha=0.9,
                    linewidths=1.5,
                    zorder=3,
                )
            budget = scene["penalty_budget"]
            penalty = np.array(scene["penalty"]).reshape(n, n)
            if budget > 0:
                ax.contourf(
                    positions,
                    positions,
                    penalty,
                    levels=[-1.0, budget],
                    colors=["#4682b4"],
                    alpha=0.13,
                    zorder=4,
                )
                ax.contour(
                    positions,
                    positions,
                    penalty,
                    levels=[budget],
                    colors=["#4682b4"],
                    linewidths=1.5,
                    zorder=4,
                )
            elif scene["penalty_zero_set"] == "plane":
                ax.axhspan(-radius, radius, color="#4682b4", alpha=0.13, zorder=4)
            elif scene["penalty_zero_set"] == "diagonal":
                ax.plot(
                    [-radius, radius],
                    [-radius, radius],
                    color="#4682b4",
                    linewidth=1.5,
                    zorder=4,
                )
            else:
                ax.plot(0, 0, marker="o", markersize=8, color="#4682b4", zorder=4)
        if field == "objective":
            points = fixture["trace"]["points"]
            coefficients = np.array([point["coefficients"] for point in points])
            ax.plot(*coefficients.T, color="#000000", linewidth=1.1, zorder=5)
            updates = np.array(
                [
                    point["coefficients"]
                    for point in points
                    if point["kind"] not in ("Initial", "Final")
                ]
            )
            if len(updates):
                ax.plot(
                    *updates.T,
                    linestyle="none",
                    marker="o",
                    markersize=3,
                    color="#000000",
                    markeredgecolor="none",
                    markeredgewidth=0,
                    zorder=5,
                )
            ax.plot(
                *coefficients[0],
                marker="s",
                markersize=4,
                color="#000000",
                markeredgecolor="none",
                markeredgewidth=0,
                zorder=5,
            )
        if scene["reference"]["converged"]:
            ax.plot(
                *scene["reference"]["coefficients"],
                marker="o",
                markersize=5,
                color="#cc6600",
                markeredgecolor="none",
                markeredgewidth=0,
                zorder=6,
            )
        ax.set(
            xlim=(-radius, radius),
            ylim=(-radius, radius),
            aspect="equal",
            title=title,
            xlabel=r"$\beta_1$",
            ylabel=r"$\beta_2$",
        )
        ax.set_xticks([-radius, 0, radius])
        ax.set_yticks([-radius, 0, radius])
        ax.set_axisbelow(True)
        ax.grid(color="#ffffff", alpha=0.12, linewidth=0.5)
        ax.axhline(0, color="#ffffff", alpha=0.2, linewidth=0.7, zorder=1)
        ax.axvline(0, color="#ffffff", alpha=0.2, linewidth=0.7, zorder=1)
        for spine in ax.spines.values():
            spine.set_color("#d9e1e6")
    for artist in figure.findobj():
        artist.set_clip_on(False)
    buffer = StringIO()
    figure.savefig(buffer, format="svg", metadata={"Date": None})
    plt.close(figure)
    svg = portable_svg(buffer.getvalue())
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text() != svg:
            raise SystemExit("Explorer fallback is stale; run task docs:widget:figure.")
    else:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(svg)


if __name__ == "__main__":
    main()
