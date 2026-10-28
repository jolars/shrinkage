# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib==3.10.7", "numpy==2.3.5"]
# ///
"""Generate deterministic SVG constraint regions for the penalty guide."""

import argparse
from collections.abc import Callable
from copy import deepcopy
from io import StringIO
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

OUTPUT = Path(__file__).resolve().parents[1] / "docs/pages/penalties/figures"
BLUE = "#2563a6"
FILL = "#d9e9f5"
ScalarTerm = Callable[[np.ndarray, np.ndarray], np.ndarray]


def l1(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.abs(x) + np.abs(y)


def l2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.hypot(x, y)


def slope(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.maximum(np.abs(x), np.abs(y)) + 0.5 * np.minimum(np.abs(x), np.abs(y))


def fusion(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 0.5 * (l1(x, y) + np.abs(x - y))


def region(ax: Axes, linear: ScalarTerm, quadratic: float = 0.0) -> None:
    # Multiples of pi/4 retain the exact corners of the polyhedral penalties.
    angles = np.linspace(0.0, 2.0 * np.pi, 1025)
    x, y = np.cos(angles), np.sin(angles)
    a = linear(x, y)
    radius = 2.0 / (a + np.sqrt(a * a + 4.0 * quadratic))
    x, y = radius * x, radius * y
    values = linear(x, y) + quadratic * (x * x + y * y)
    if not np.allclose(values, 1.0, rtol=0.0, atol=2e-14):
        raise AssertionError("A plotted boundary does not satisfy P(theta) = 1.")
    if not np.all(linear(0.5 * x, 0.5 * y) + quadratic * 0.25 * (x * x + y * y) < 1):
        raise AssertionError("Interior samples must be feasible.")
    if not np.all(linear(1.5 * x, 1.5 * y) + quadratic * 2.25 * (x * x + y * y) > 1):
        raise AssertionError("Exterior samples must be infeasible.")
    ax.fill(x, y, color=FILL, zorder=2)
    ax.plot(x, y, color=BLUE, linewidth=2, zorder=3)


def strip(ax: Axes, diagonal: bool) -> None:
    x = np.linspace(-2.0, 2.0, 5)
    center = x if diagonal else np.zeros_like(x)
    ax.fill_between(
        x,
        np.clip(center - 1.0, -2.0, 2.0),
        np.clip(center + 1.0, -2.0, 2.0),
        color=FILL,
        zorder=2,
    )
    for offset in (-1.0, 1.0):
        if diagonal:
            line_x = np.array([max(-2.0, -2.0 - offset), min(2.0, 2.0 - offset)])
            ax.plot(line_x, line_x + offset, color=BLUE, linewidth=2, zorder=3)
        else:
            ax.plot(x, center + offset, color=BLUE, linewidth=2, zorder=3)
    for direction in (-1, 1):
        start = (direction * 1.2, direction * 1.2 if diagonal else 0.0)
        end = (direction * 1.8, direction * 1.8 if diagonal else 0.0)
        ax.annotate(
            "", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": BLUE}
        )
    ax.text(0.0, -1.8, "Unbounded", ha="center", fontsize=10, color=BLUE)


def panel(ax: Axes, title: str) -> None:
    ax.set(xlim=(-2.0, 2.0), ylim=(-2.0, 2.0), aspect="equal", title=title)
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_xlabel(r"$\theta_1$", fontsize=13)
    ax.set_ylabel(r"$\theta_2$", fontsize=13, rotation=0, labelpad=8)
    ax.grid(color="#e5e7eb", linewidth=0.6, zorder=0)
    ax.axhline(0, color="#9ca3af", linewidth=0.8, zorder=1)
    ax.axvline(0, color="#9ca3af", linewidth=0.8, zorder=1)
    for spine in ax.spines.values():
        spine.set_color("#c5cbd3")


def portable_svg(source: str) -> str:
    # Diplodocus accepts self-contained geometry rather than CSS or SVG references.
    namespace = "http://www.w3.org/2000/svg"
    ET.register_namespace("", namespace)
    tree = ET.fromstring(source)
    definitions = {
        node.attrib["id"]: node for node in tree.iter() if "id" in node.attrib
    }

    def expand(node: ET.Element) -> ET.Element | None:
        tag = node.tag.rsplit("}", 1)[-1]
        if tag in {"metadata", "defs", "style"}:
            return None
        if tag == "use":
            reference = node.attrib["{http://www.w3.org/1999/xlink}href"]
            path = deepcopy(definitions[reference.removeprefix("#")])
            group = ET.Element(f"{{{namespace}}}g")
            for key in ("transform", "style"):
                if key in node.attrib:
                    group.set(key, node.attrib[key])
            translation = f"translate({node.get('x', '0')} {node.get('y', '0')})"
            group.set(
                "transform", f"{group.get('transform', '')} {translation}".strip()
            )
            group.append(path)
            return expand(group)
        node.attrib.pop("id", None)
        node.attrib.pop("version", None)
        style = node.attrib.pop("style", "")
        for declaration in style.split(";"):
            if declaration.strip():
                key, value = declaration.split(":", 1)
                node.set(key.strip(), value.strip())
        children = [
            converted for child in node if (converted := expand(child)) is not None
        ]
        node[:] = children
        return node

    result = expand(tree)
    if result is None:
        raise AssertionError("Missing SVG root.")
    result.set("stroke-linejoin", "round")
    result.set("stroke-linecap", "butt")
    result.set("role", "img")
    return ET.tostring(result, encoding="unicode") + "\n"


def render(name: str) -> str:
    titles = {
        "l1": ["Lasso"],
        "ridge": ["Ridge"],
        "elastic-net": ["Elastic net"],
        "slope": ["SLOPE"],
        "group-lasso": ["One group", "Two singleton groups"],
        "clustered-lasso": ["Sparsity and fusion", "Pure fusion"],
        "chain-fused-lasso": ["Sparsity and adjacent fusion", "Pure fusion"],
        "scaled": ["L1", "Twice L1"],
        "masked": ["First coordinate unpenalized"],
        "axis": ["Features: separate responses", "Responses: joint response vector"],
    }[name]
    fig, axes = plt.subplots(
        len(titles), 1, figsize=(4.2, 4.0 * len(titles)), layout="constrained"
    )
    axes = np.atleast_1d(axes)
    for ax, title in zip(axes, titles, strict=True):
        panel(ax, title)
    match name:
        case "l1":
            region(axes[0], l1)
        case "ridge":
            region(axes[0], lambda x, y: np.zeros_like(x), quadratic=1.0)
        case "elastic-net":
            region(axes[0], lambda x, y: 0.5 * l1(x, y), quadratic=0.5)
        case "slope":
            region(axes[0], slope)
        case "group-lasso":
            region(axes[0], l2)
            region(axes[1], l1)
        case "clustered-lasso" | "chain-fused-lasso":
            region(axes[0], fusion)
            strip(axes[1], diagonal=True)
        case "scaled":
            region(axes[0], l1)
            region(axes[1], lambda x, y: 2 * l1(x, y))
        case "masked":
            strip(axes[0], diagonal=False)
        case "axis":
            region(axes[0], l1)
            region(axes[1], l2)
    for artist in fig.findobj():
        artist.set_clip_on(False)
    output = StringIO()
    fig.savefig(
        output,
        format="svg",
        metadata={"Date": None, "Title": "Penalty constraint region"},
    )
    plt.close(fig)
    return portable_svg(output.getvalue())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Check committed figures without writing."
    )
    args = parser.parse_args()
    plt.rcParams.update(
        {"svg.hashsalt": "shrinkage-penalty-regions", "font.family": "DejaVu Sans"}
    )
    if not args.check:
        OUTPUT.mkdir(parents=True, exist_ok=True)
    for name in (
        "l1",
        "ridge",
        "elastic-net",
        "slope",
        "group-lasso",
        "clustered-lasso",
        "chain-fused-lasso",
        "scaled",
        "masked",
        "axis",
    ):
        content = render(name)
        path = OUTPUT / f"{name}.svg"
        if args.check:
            if not path.is_file() or path.read_text() != content:
                raise SystemExit(
                    f"Regenerate {path.relative_to(OUTPUT.parents[4])} with task docs:figures."
                )
        else:
            path.write_text(content)
    print("Checked" if args.check else "Generated", "10 penalty constraint figures.")


if __name__ == "__main__":
    main()
