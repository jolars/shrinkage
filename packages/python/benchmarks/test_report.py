"""Keep convergence points distinct from metric rows and cumulative run time."""

from copy import deepcopy

import pytest
from report import collect_observations, convergence_points


def exported_rows():
    rows = []
    for method in ("shrinkage", "scikit-learn"):
        for repetition in range(5):
            for budget in (1, 2, 3):
                for metric, value in (
                    ("relative_gap", 10.0 ** (-3 * budget)),
                    ("duality_gap", 2 * 10.0 ** (-3 * budget)),
                    ("objective", 0.625),
                ):
                    rows.append(
                        {
                            "record_type": "observation",
                            "id": f"{method}-{repetition}-{budget}",
                            "attempt_id": f"attempt-{method}-{repetition}-{budget}",
                            "warmup": False,
                            "valid": True,
                            "completion": "reached" if budget == 3 else "not_reached",
                            "profile": {
                                "delivery": "fresh_sequence",
                                "measurement_index": repetition,
                                "schedule_index": budget - 1,
                                "observation_control": {"iterations": budget},
                            },
                            "compatibility": {"timing_scope": "execute_only"},
                            "provenance": {
                                "data": {
                                    "implementation_statistics": {
                                        "implementation": method,
                                        "version": "0.0.0",
                                        "iterations": budget,
                                        "threadpools": [{"num_threads": 1}],
                                    }
                                }
                            },
                            "implementation_time_ns": budget * 1000,
                            "timed_wall_time_ns": budget * 2000,
                            "metric_name": f"/{metric}",
                            "metric_value": value,
                        }
                    )
    return rows


def test_group_metrics_by_observation_and_exclude_warmups():
    rows = exported_rows()
    warmup = deepcopy(rows[0])
    warmup.update(id="warmup", warmup=True)
    observations = collect_observations([*rows, warmup])
    assert len(observations) == 30
    assert all(len(row["metrics"]) == 3 for row in observations)
    assert {row["iteration_limit"] for row in observations} == {1, 2, 3}
    assert {row["repetition"] for row in observations} == set(range(5))


def test_reject_partial_comparison():
    with pytest.raises(ValueError, match="did not reach the accuracy target"):
        collect_observations(exported_rows()[:-3])


def test_curve_uses_independent_fit_times_and_absolute_duality_gaps():
    snapshot = {"observations": collect_observations(exported_rows())}
    points = convergence_points(snapshot, "shrinkage")
    assert [point["time_ms"] for point in points] == pytest.approx(
        [0.001, 0.002, 0.003]
    )
    assert [point["duality_gap"] for point in points] == pytest.approx(
        [2e-3, 2e-6, 2e-9]
    )


def test_reject_invalid_solution():
    rows = exported_rows()
    rows[0]["valid"] = False
    with pytest.raises(ValueError, match="invalid"):
        collect_observations(rows)


def test_reject_missing_repetition():
    rows = [row for row in exported_rows() if not row["id"].startswith("shrinkage-4-")]
    with pytest.raises(ValueError, match="Expected 5 curves"):
        collect_observations(rows)


def test_reject_failed_attempt_even_with_complete_observations():
    failure = {"record_type": "attempt", "failure_code": "timeout"}
    with pytest.raises(ValueError, match="Failed attempt: timeout"):
        collect_observations([*exported_rows(), failure])
