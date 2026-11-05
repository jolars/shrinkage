# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Regenerate with `uv run --script generate_probit.py > probit.csv`.

Independent 100-digit normal-tail and penalized-fit references. The fit oracle
solves full original-loss stationarity equations rather than Fisher-scoring
subproblems. Fixed positive slope signs and an explicit inactive KKT check
identify the lasso solution for this small fixture.
"""

import csv
import sys

import mpmath as mp

mp.mp.dps = 100
out = csv.writer(sys.stdout, lineterminator="\n")
out.writerow(["kind", "a", "b", "loss", "gradient", "hessian", "fisher"])


def row(eta, y):
    p = mp.erfc(-eta / mp.sqrt(2)) / 2
    q = mp.erfc(eta / mp.sqrt(2)) / 2
    log_p = mp.log(p) if eta <= 0 else mp.log1p(-q)
    log_q = mp.log(q) if eta >= 0 else mp.log1p(-p)
    phi = mp.exp(-(eta**2) / 2) / mp.sqrt(2 * mp.pi)
    gradient = phi * ((1 - y) / q - y / p)
    hessian = phi**2 * (y / p**2 + (1 - y) / q**2) - eta * gradient
    return -y * log_p - (1 - y) * log_q, gradient, hessian, phi**2 / (p * q)


def emit(values):
    out.writerow([mp.nstr(v, 40) if isinstance(v, mp.mpf) else v for v in values])


for eta in map(mp.mpf, [-40, -12, -2, 0, 2, 12, 40]):
    for y in map(mp.mpf, ["0", "0.3", "1"]):
        emit(["tail", eta, y, *row(eta, y)])

x = list(map(mp.mpf, ["-2", "-1", "0", "0.5", "1", "2"]))
y = list(map(mp.mpf, ["0.05", "0.2", "0.3", "0.65", "0.8", "0.95"]))


def evaluate(intercept, beta, l1, l2):
    rows = [row(intercept + beta * a, b) for a, b in zip(x, y, strict=True)]
    loss = sum(r[0] for r in rows) / len(x) + l1 * abs(beta) + l2 * beta**2 / 2
    intercept_gradient = sum(r[1] for r in rows) / len(x)
    slope_gradient = (
        sum(a * r[1] for a, r in zip(x, rows, strict=True)) / len(x) + l2 * beta
    )
    return loss, intercept_gradient, slope_gradient


def fit_case(l1, l2):
    l1, l2 = mp.mpf(l1), mp.mpf(l2)
    intercept = mp.findroot(lambda a: evaluate(a, 0, l1, l2)[1], 0, tol=mp.mpf("1e-90"))
    beta = mp.mpf(0)
    if abs(evaluate(intercept, beta, l1, l2)[2]) > l1:
        intercept, beta = mp.findroot(
            lambda a, b: (evaluate(a, b, l1, l2)[1], evaluate(a, b, l1, l2)[2] + l1),
            (0, mp.mpf("0.7")),
            tol=mp.mpf("1e-90"),
        )
        assert beta > 0
    loss, g0, g = evaluate(intercept, beta, l1, l2)
    assert abs(g0) < mp.mpf("1e-85")
    assert abs(g + l1) < mp.mpf("1e-85") if beta else abs(g) <= l1
    emit(["fit", l1, l2, loss, intercept, beta, mp.mpf(0)])


for l1, l2 in [("0", "0.2"), ("0.05", "0"), ("0.03", "0.1"), ("1", "0.1")]:
    fit_case(l1, l2)
