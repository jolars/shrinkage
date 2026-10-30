"""Check figure-specific assumptions using Metewand analysis observations."""

from copy import deepcopy

import pytest
from report import collect_observations, convergence_points


def snapshot():
    observations = []
    for method in ("shrinkage", "scikit-learn"):
        for repetition in range(3):
            for budget in (1, 2, 3):
                observations.append(
                    {
                        "observation_id": f"{method}-{repetition}-{budget}",
                        "implementation": method,
                        "selected": True,
                        "warmup": False,
                        "valid": True,
                        "completion": "reached" if budget == 3 else "not_reached",
                        "measurement_index": repetition,
                        "timing_scope": "execute_only",
                        "profile": {
                            "delivery": "fresh_sequence",
                            "schedule_index": budget - 1,
                            "observation_control": {"iterations": budget},
                        },
                        "statistics": {
                            "version": "0.0.0",
                            "threadpools": [{"num_threads": 1}],
                        },
                        "provenance_id": "shared",
                        "implementation_time_ns": budget * 1000,
                        "timed_wall_time_ns": budget * 2000,
                        "metrics": {
                            "relative_gap": 10.0 ** (-3 * budget),
                            "duality_gap": 2 * 10.0 ** (-3 * budget),
                        },
                    }
                )
    return {
        "schema_version": 1,
        "completeness": {"complete": True, "issues": []},
        "candidates": [
            {"implementation": method, "measurement_repetitions": 3}
            for method in ("shrinkage", "scikit-learn")
        ],
        "observations": observations,
        "attempts": [],
        "provenance": {"shared": {"dataset": {"output": {"sha256": "same"}}}},
    }


def test_selected_observations_keep_intermediate_points_and_exclude_warmups():
    result = snapshot()
    warmup = deepcopy(result["observations"][0])
    warmup["warmup"] = True
    superseded = deepcopy(warmup)
    superseded.update(warmup=False, selected=False, valid=False)
    result["observations"].extend([warmup, superseded])
    observations = collect_observations(result)
    assert len(observations) == 18
    assert any(row["completion"] == "not_reached" for row in observations)


def test_reject_incomplete_export():
    result = snapshot()
    result["completeness"]["complete"] = False
    with pytest.raises(ValueError, match="incomplete"):
        collect_observations(result)


def test_curve_uses_each_fit_timer_and_declared_repetition_count():
    points = convergence_points(snapshot(), "shrinkage")
    assert [point["time_ms"] for point in points] == pytest.approx(
        [0.001, 0.002, 0.003]
    )
    assert [point["duality_gap"] for point in points] == pytest.approx(
        [2e-3, 2e-6, 2e-9]
    )


def test_reject_control_without_enough_measurements_for_error_bars():
    result = snapshot()
    result["observations"].pop()
    with pytest.raises(ValueError, match="declared repetitions"):
        convergence_points(result, "scikit-learn")


def test_successful_retry_can_follow_a_failed_attempt():
    result = snapshot()
    result["attempts"].append({"state": "failed", "failure_code": "timeout"})
    assert len(collect_observations(result)) == 18


def test_reject_multiple_datasets():
    result = snapshot()
    result["provenance"]["different"] = {"dataset": {"output": {"sha256": "other"}}}
    result["observations"][0]["provenance_id"] = "different"
    with pytest.raises(ValueError, match="same dataset"):
        collect_observations(result)


def test_reject_multiple_threads():
    result = snapshot()
    result["observations"][0]["statistics"]["threadpools"][0]["num_threads"] = 2
    with pytest.raises(ValueError, match="one thread"):
        collect_observations(result)
