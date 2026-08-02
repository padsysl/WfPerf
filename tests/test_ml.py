# Copyright (c) 2026, University of Florida. All rights reserved.
#
# This file is part of WfPerf, developed by members of Prof. Xiaoyi Lu's
# group at the University of Florida. See LICENSE in the top-level directory.

import csv
import json
from argparse import Namespace

import pytest

from wfperf.cli import _process_ml
from wfperf.config import parse_workflow
from wfperf.ml.data import (
    DATABASE_COLUMNS,
    FEATURE_NAMES,
    TARGET_NAMES,
    append_observation,
    extract_features,
    extract_targets,
)
from wfperf.ml.inference import MODEL_SCHEMA_VERSION, validate_run
from wfperf.ml.training import load_database, train
from wfperf.result import RunResult


class FixedEstimator:
    def __init__(self, values):
        self.values = values

    def predict(self, features):
        return [self.values for _ in features]


def workflow():
    return parse_workflow(
        {
            "tasks": [
                {
                    "func": "source",
                    "count": 2,
                    "nprocs": 3,
                    "sleep": 4,
                    "iter": 5,
                    "particle": 6,
                },
                {
                    "func": "intermediate",
                    "number of intermediate": 2,
                    "count": [1, 3],
                    "nprocs": [2, 4],
                    "sleep": [10, 20],
                    "iter": [2, 6],
                    "particle": [100, 300],
                },
                {
                    "func": "sink",
                    "count": 1,
                    "nprocs": 7,
                    "sleep": 8,
                    "iter": 9,
                },
            ]
        }
    )


def summary():
    return {
        "backend": "parsl",
        "e2e_time_seconds": 9.0,
        "tasks": [
            {
                "role": "source",
                "task_time_ms": 10.0,
                "input_time_ms": 0.0,
                "output_time_ms": 2.0,
            },
            {
                "role": "source",
                "task_time_ms": 14.0,
                "input_time_ms": 0.0,
                "output_time_ms": 4.0,
            },
            {
                "role": "intermediate",
                "task_time_ms": 20.0,
                "input_time_ms": 5.0,
                "output_time_ms": 6.0,
            },
            {
                "role": "intermediate",
                "task_time_ms": 30.0,
                "input_time_ms": 7.0,
                "output_time_ms": 8.0,
            },
            {
                "role": "sink",
                "task_time_ms": 40.0,
                "input_time_ms": 9.0,
                "output_time_ms": 0.0,
            },
        ],
    }


def test_extracts_backend_independent_features_and_targets():
    features = extract_features(workflow())
    targets = extract_targets(summary())

    assert list(features) == list(FEATURE_NAMES)
    assert features["source_particles_per_process"] == 6
    assert features["intermediate_count"] == 4
    assert features["intermediate_nprocs"] == 3.5
    assert features["intermediate_sleep_seconds"] == 17.5
    assert features["intermediate_particles_per_process"] == 250
    assert targets["source_task_time_ms"] == 12
    assert targets["intermediate_task_time_ms"] == 25
    assert targets["sink_input_time_ms"] == 9


def test_appends_canonical_database_and_rejects_other_schema(tmp_path):
    path = tmp_path / "runs.csv"
    features = extract_features(workflow())
    targets = extract_targets(summary())
    append_observation(path, features, targets)
    append_observation(path, features, targets)

    with path.open(newline="") as stream:
        rows = list(csv.reader(stream))
    assert tuple(rows[0]) == DATABASE_COLUMNS
    assert len(rows) == 3

    incompatible = tmp_path / "other.csv"
    incompatible.write_text("not,a,wfperf,database\n")
    with pytest.raises(ValueError, match="incompatible"):
        append_observation(incompatible, features, targets)


def test_process_records_run_when_model_is_not_available(tmp_path):
    run_summary = summary()
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(run_summary))
    result = RunResult(tmp_path, summary_path, run_summary)
    args = Namespace(
        ml_database=tmp_path / "runs.csv",
        ml_model=tmp_path / "missing.joblib",
        ml_threshold=5.0,
        no_ml_record=False,
        no_ml_validation=False,
    )

    _process_ml(workflow(), result, args)

    saved = json.loads(summary_path.read_text())
    assert saved["ml"]["recorded"] is True
    assert saved["ml"]["validation"]["status"] == "model_not_found"


def test_validation_reports_only_differences_above_threshold(tmp_path, capsys):
    joblib = pytest.importorskip("joblib")
    values = [12.0, 25.0, 40.0, 3.0, 6.0, 7.0, 9.0, 8.0]
    artifact = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "algorithm": "fixed-test-model",
        "feature_names": FEATURE_NAMES,
        "target_names": TARGET_NAMES,
        "estimator": FixedEstimator(values),
    }
    model_path = tmp_path / "model.joblib"
    joblib.dump(artifact, model_path)

    validation = validate_run(
        model_path,
        extract_features(workflow()),
        extract_targets(summary()),
        threshold_percent=5.0,
    )
    assert validation["status"] == "warning"
    assert [item["metric"] for item in validation["violations"]] == [
        "e2e_time_seconds"
    ]
    assert validation["difference_percent"]["e2e_time_seconds"] == pytest.approx(
        100.0 / 9.0
    )

    run_summary = summary()
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(run_summary))
    result = RunResult(tmp_path, summary_path, run_summary)
    args = Namespace(
        ml_database=tmp_path / "runs.csv",
        ml_model=model_path,
        ml_threshold=5.0,
        no_ml_record=True,
        no_ml_validation=False,
    )
    _process_ml(workflow(), result, args)
    assert "WfPerf ML warning: e2e_time_seconds" in capsys.readouterr().err


def test_loads_legacy_database_and_trains_offline(tmp_path):
    pytest.importorskip("sklearn")
    database = tmp_path / "legacy.csv"
    legacy_header = [
        "count", "nprocs", "sleep", "particle", "iter",
        "count", "nprocs", "sleep", "particle", "iter",
        "count", "nprocs", "sleep", "iter",
        "producer_time", "middle_time", "consumer_time", "producer_output",
        "middle_input", "middle_output", "consumer_input", "e2e_time",
    ]
    with database.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(legacy_header)
        for index in range(1, 13):
            features = [float(index + offset) for offset in range(14)]
            targets = [float(index * (offset + 1)) for offset in range(8)]
            writer.writerow(features + targets)

    features, targets = load_database(database)
    assert features.shape == (12, 14)
    assert targets.shape == (12, 8)

    output = tmp_path / "trained.joblib"
    artifact = train(
        database,
        output,
        algorithm="polynomial-linear",
        folds=3,
    )
    assert output.is_file()
    assert artifact["algorithm"] == "polynomial-linear"
    assert artifact["training_rows"] == 12
