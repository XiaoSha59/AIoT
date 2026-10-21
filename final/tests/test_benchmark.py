"""Numerical reporting/provenance tests; these fixtures never train a model."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from training.benchmark_runtime import classification_metrics, summary, worker


def test_latency_percentiles_and_empty_input():
    result = summary([1., 2., 3., 4.])
    assert result["samples"] == 4
    assert result["mean_ms"] == result["p50_ms"] == 2.5
    assert result["p95_ms"] == pytest.approx(3.85)
    with pytest.raises(ValueError, match="No measurements"):
        summary([])


def test_metrics_include_threshold_rejection_and_stable_loss():
    result = classification_metrics(np.array([[3., 0.], [0., 3.], [0., 1.]]),
                                    np.array([0, 1, 1]), ["Unknown", "son"], .8)
    assert result["accuracy"] == pytest.approx(2/3)
    assert result["known_member_accuracy"] == .5
    assert result["unknown_rejection_rate"] == 1
    assert result["false_acceptance_rate"] == 0
    assert result["macro_f1"] == pytest.approx(2/3)
    assert result["confusion_matrix"] == [[1, 0], [1, 1]]
    stable = classification_metrics(np.array([[1000., 0.], [0., 1000.]]),
                                    np.array([0, 1]), ["Unknown", "son"], .8)
    assert stable["test_cross_entropy_unweighted"] == 0


def test_benchmark_refuses_calibration_from_a_different_manifest(tmp_path):
    data, model = tmp_path/"data", tmp_path/"model"
    data.mkdir()
    model.mkdir()
    (data/"manifest.json").write_text(json.dumps({"classes": ["Unknown", "son"]}))
    (model/"threshold.json").write_text(json.dumps({"source": "validation", "manifest_sha256": "wrong"}))
    (model/"export_validation.json").write_text("{}")
    with pytest.raises(ValueError, match="provenance mismatch"):
        worker(SimpleNamespace(data=data, model=model))
