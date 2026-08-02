# Copyright (c) 2026, University of Florida. All rights reserved.
#
# This file is part of WfPerf, developed by members of Prof. Xiaoyi Lu's
# group at the University of Florida. See LICENSE in the top-level directory.

"""Load a locally trained WfPerf model and validate one completed run."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

from wfperf.ml.data import FEATURE_NAMES, TARGET_NAMES


MODEL_SCHEMA_VERSION = 1


def load_model(path: Path) -> Mapping[str, Any]:
    """Load and validate a trusted, locally generated WfPerf model artifact."""

    try:
        import joblib
    except ImportError as error:
        raise ImportError("ML validation requires the 'ml' installation extra") from error

    artifact = joblib.load(path)
    if not isinstance(artifact, Mapping):
        raise ValueError("model artifact is not a WfPerf model")
    if artifact.get("schema_version") != MODEL_SCHEMA_VERSION:
        raise ValueError("unsupported WfPerf model schema")
    if tuple(artifact.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("model feature schema does not match this WfPerf version")
    if tuple(artifact.get("target_names", ())) != TARGET_NAMES:
        raise ValueError("model target schema does not match this WfPerf version")
    if "estimator" not in artifact:
        raise ValueError("model artifact does not contain an estimator")
    return artifact


def validate_run(
    model_path: Path,
    features: Mapping[str, float],
    targets: Mapping[str, float],
    threshold_percent: float = 5.0,
) -> Dict[str, Any]:
    """Predict all timings and report measurements outside the threshold."""

    if threshold_percent < 0:
        raise ValueError("ML difference threshold must be non-negative")

    import numpy as np

    artifact = load_model(model_path)
    inputs = np.asarray([[features[name] for name in FEATURE_NAMES]], dtype=float)
    prediction = np.asarray(artifact["estimator"].predict(inputs), dtype=float)
    if prediction.shape != (1, len(TARGET_NAMES)):
        raise ValueError("model returned an unexpected prediction shape")
    prediction = np.maximum(prediction[0], 0.0)

    predicted = {}
    actual = {}
    differences = {}
    violations = []
    for index, name in enumerate(TARGET_NAMES):
        expected = float(prediction[index])
        measured = float(targets[name])
        if measured == 0.0:
            difference = 0.0 if expected == 0.0 else None
        else:
            difference = abs(measured - expected) / abs(measured) * 100.0
        predicted[name] = expected
        actual[name] = measured
        differences[name] = difference
        if difference is None or difference > threshold_percent:
            violations.append(
                {
                    "metric": name,
                    "actual": measured,
                    "predicted": expected,
                    "difference_percent": difference,
                }
            )

    return {
        "status": "warning" if violations else "within_threshold",
        "model": str(model_path),
        "algorithm": artifact.get("algorithm", "unknown"),
        "threshold_percent": float(threshold_percent),
        "predicted": predicted,
        "actual": actual,
        "difference_percent": differences,
        "violations": violations,
    }
