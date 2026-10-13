"""Regenerate with `python3 generate_binomial_decimal.py > binomial_decimal.csv`.

This independent reference uses 110-digit Decimal arithmetic, damped Newton
steps, and exhaustive active-set/sign enumeration rather than proximal gradient.
It minimizes mean Bernoulli negative log likelihood + l1*||b||_1 + l2*||b||^2/2.
Every emitted solution satisfies coefficient and unpenalized-intercept KKT
conditions to 1e-45 on the optimization scale. No third-party packages are needed.
"""

import csv
import sys
from decimal import Decimal as D
from decimal import getcontext
from itertools import product

getcontext().prec = 110
ZERO, ONE = D(0), D(1)
X = [
    [D(v) for v in row]
    for row in [
        [0, 1, 0],
        [1, 0, 2],
        [2, 2, 0],
        [3, -1, 1],
        [4, 3, 0],
        [5, 1, 2],
        [6, 4, 1],
        [7, -2, 0],
    ]
]
EXTREME_X = [
    [D(v) for v in row]
    for row in [
        [-40, 1, 0],
        [-20, 0, 2],
        [-1, 2, 0],
        ["-0.5", -1, 1],
        ["0.5", 3, 0],
        [1, 1, 2],
        [20, 4, 1],
        [40, -2, 0],
    ]
]
Y = list(map(D, [0, 0, 0, 1, 1, 0, 1, 1]))
EXTREME_Y = list(map(D, [0, 0, 0, 0, 1, 1, 1, 1]))
FRACTIONAL_Y = list(
    map(D, ["0.05", "0.2", "0.1", "0.9", "0.75", "0.35", "0.8", "0.95"])
)
PREDICT_X = [
    [D(v) for v in row]
    for row in [
        ["1.5", "-0.5", 1],
        [8, 2, 0],
        [-1000, 1, 2],
        [1000, -1, 0],
    ]
]


def dot(a, b):
    return sum((u * v for u, v in zip(a, b)), ZERO)


def evaluate(design, y, coefficients, l1, l2, signs):
    p, n = len(coefficients), D(len(y))
    value = ZERO
    gradient = [ZERO] * p
    hessian = [[ZERO] * p for _ in range(p)]
    for row, target in zip(design, y):
        eta = dot(row, coefficients)
        tail = (-abs(eta)).exp()
        residual = (
            (ONE - target) - tail / (ONE + tail)
            if eta >= 0
            else tail / (ONE + tail) - target
        )
        value += (max(eta, ZERO) - target * eta + (ONE + tail).ln()) / n
        curvature = tail / (ONE + tail) ** 2 / n
        for j in range(p):
            gradient[j] += row[j] * residual / n
            for k in range(p):
                hessian[j][k] += row[j] * row[k] * curvature
    for j, sign in enumerate(signs):
        value += l1 * sign * coefficients[j] + l2 * coefficients[j] ** 2 / 2
        gradient[j] += l1 * sign + l2 * coefficients[j]
        hessian[j][j] += l2
    return value, gradient, hessian


def solve(matrix, rhs):
    augmented = [row[:] + [value] for row, value in zip(matrix, rhs)]
    for j in range(len(rhs)):
        pivot = max(range(j, len(rhs)), key=lambda k: abs(augmented[k][j]))
        augmented[j], augmented[pivot] = augmented[pivot], augmented[j]
        divisor = augmented[j][j]
        if abs(divisor) < D("1e-60"):
            raise ArithmeticError("singular Newton system")
        augmented[j] = [v / divisor for v in augmented[j]]
        for k in range(len(rhs)):
            if k != j:
                factor = augmented[k][j]
                augmented[k] = [
                    a - factor * b for a, b in zip(augmented[k], augmented[j])
                ]
    return [row[-1] for row in augmented]


def reference(x, y, normalization, intercept, l1, l2):
    p = len(x[0])
    means = [sum(row[j] for row in x) / len(x) for j in range(p)]
    centers = means if normalization == "auto" and intercept else [ZERO] * p
    scales = (
        [(sum((row[j] - means[j]) ** 2 for row in x) / len(x)).sqrt() for j in range(p)]
        if normalization == "auto"
        else [ONE] * p
    )
    design = [
        [(v - center) / scale for v, center, scale in zip(row, centers, scales)]
        + ([ONE] if intercept else [])
        for row in x
    ]
    for signs in product((-1, 0, 1), repeat=p) if l1 else [(1,) * p]:
        active = [j for j, sign in enumerate(signs) if sign]
        indices = active + ([p] if intercept else [])
        reduced = [[row[j] for j in indices] for row in design]
        active_signs = [D(signs[j]) for j in active]
        coefficients = [ZERO] * len(indices)
        for _ in range(200):
            value, gradient, hessian = evaluate(
                reduced, y, coefficients, l1, l2, active_signs
            )
            if max(map(abs, gradient), default=ZERO) < D("1e-50"):
                break
            try:
                direction = solve(hessian, gradient)
            except ArithmeticError:
                # Some signed subproblems are unbounded; KKT checks reject them.
                break
            step = ONE
            for _ in range(200):
                candidate = [a - step * b for a, b in zip(coefficients, direction)]
                trial = evaluate(reduced, y, candidate, l1, l2, active_signs)[0]
                if trial <= value - D("0.0001") * step * dot(gradient, direction):
                    coefficients = candidate
                    break
                step /= 2
            else:
                break
        else:
            continue
        full = [ZERO] * (p + int(intercept))
        for j, coefficient in zip(indices, coefficients):
            full[j] = coefficient
        if l1 and any(full[j] * signs[j] <= ZERO for j in active):
            continue
        _, smooth_gradient, _ = evaluate(design, y, full, ZERO, l2, [ONE] * p)
        residuals = [
            abs(smooth_gradient[j] + l1 * signs[j])
            if signs[j]
            else max(abs(smooth_gradient[j]) - l1, ZERO)
            for j in range(p)
        ]
        if intercept:
            residuals.append(abs(smooth_gradient[p]))
        if max(residuals) > D("1e-45"):
            continue
        objective = evaluate(design, y, full, ZERO, l2, [ONE] * p)[0] + l1 * sum(
            map(abs, full[:p])
        )
        original = [b / scale for b, scale in zip(full, scales)]
        offset = (full[p] if intercept else ZERO) - dot(original, centers)
        predictions = [dot(row, original) + offset for row in PREDICT_X]
        max_logit = max(abs(dot(row, full)) for row in design)
        return [offset] + original + [objective, max_logit] + predictions
    raise ArithmeticError("no active set satisfies KKT conditions")


def main():
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(
        [
            (
                "# Generated with 110-digit Decimal Newton and active-set enumeration; "
                "python3 generate_binomial_decimal.py > binomial_decimal.csv"
            )
        ]
    )
    writer.writerow(
        ["case", "data", "normalization", "fit_intercept", "l1", "l2", "intercept"]
        + [f"coefficient_{j}" for j in range(3)]
        + ["objective", "max_training_logit"]
        + [f"prediction_{j}" for j in range(4)]
    )
    cases = [
        ("ridge_raw", "regular", "none", True, "0", "0.2"),
        ("ridge_raw_no_intercept", "regular", "none", False, "0", "0.2"),
        ("ridge_auto", "regular", "auto", True, "0", "0.2"),
        ("ridge_auto_no_intercept", "regular", "auto", False, "0", "0.2"),
        ("lasso_raw", "regular", "none", True, "0.03", "0"),
        ("lasso_auto", "regular", "auto", True, "0.03", "0"),
        ("elastic_net_auto", "regular", "auto", True, "0.03", "0.1"),
        ("null_model", "regular", "auto", True, "1", "0"),
        ("fractional_ridge", "fractional", "auto", True, "0", "0.2"),
        ("extreme_ridge", "extreme", "none", True, "0", "0.02"),
    ]
    for name, data, normalization, intercept, l1, l2 in cases:
        x, y = (EXTREME_X, EXTREME_Y) if data == "extreme" else (X, Y)
        if data == "fractional":
            y = FRACTIONAL_Y
        values = reference(x, y, normalization, intercept, D(l1), D(l2))
        writer.writerow(
            [name, data, normalization, int(intercept), l1, l2]
            + [format(float(v), ".17g") for v in values]
        )


if __name__ == "__main__":
    main()
