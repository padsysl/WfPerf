# Copyright (c) 2026, University of Florida. All rights reserved.
#
# This file is part of WfPerf, developed by members of Prof. Xiaoyi Lu's
# group at the University of Florida. See LICENSE in the top-level directory.

"""Stable data schema and observation storage for WfPerf performance models."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Mapping, Sequence

from wfperf.config import StageSpec, WorkflowSpec


FEATURE_NAMES = (
    "source_count",
    "source_nprocs",
    "source_sleep_seconds",
    "source_particles_per_process",
    "source_iterations",
    "intermediate_count",
    "intermediate_nprocs",
    "intermediate_sleep_seconds",
    "intermediate_particles_per_process",
    "intermediate_iterations",
    "sink_count",
    "sink_nprocs",
    "sink_sleep_seconds",
    "sink_iterations",
)

TARGET_NAMES = (
    "source_task_time_ms",
    "intermediate_task_time_ms",
    "sink_task_time_ms",
    "source_output_time_ms",
    "intermediate_input_time_ms",
    "intermediate_output_time_ms",
    "sink_input_time_ms",
    "e2e_time_seconds",
)

DATABASE_COLUMNS = FEATURE_NAMES + TARGET_NAMES


def data_directory() -> Path:
    """Return the per-user WfPerf data directory."""

    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    if xdg_data_home:
        return Path(xdg_data_home) / "wfperf"
    return Path.home() / ".local" / "share" / "wfperf"


def default_database_path() -> Path:
    """Return the observation database path, honoring its environment override."""

    value = os.environ.get("WFPERF_ML_DATABASE")
    return Path(value) if value else data_directory() / "runs.csv"


def default_model_path() -> Path:
    """Return the trained model path, honoring its environment override."""

    value = os.environ.get("WFPERF_ML_MODEL")
    return Path(value) if value else data_directory() / "model.joblib"


def _stage_features(stage: StageSpec) -> List[float]:
    return [
        float(stage.count),
        float(stage.nprocs),
        float(stage.sleep_seconds),
        float(stage.particles_per_process or 0),
        float(stage.iterations),
    ]


def _weighted_intermediate_features(stages: Sequence[StageSpec]) -> List[float]:
    """Collapse any number of intermediate stages into the legacy five fields."""

    task_count = sum(stage.count for stage in stages)
    if not task_count:
        return [0.0] * 5

    def task_weighted(attribute: str) -> float:
        return sum(
            float(getattr(stage, attribute) or 0) * stage.count for stage in stages
        ) / task_count

    return [
        float(task_count),
        task_weighted("nprocs"),
        task_weighted("sleep_seconds"),
        task_weighted("particles_per_process"),
        task_weighted("iterations"),
    ]


def extract_features(workflow: WorkflowSpec) -> Dict[str, float]:
    """Extract the model's backend-independent input values."""

    source = _stage_features(workflow.source)
    intermediate = _weighted_intermediate_features(workflow.intermediates)
    sink = [
        float(workflow.sink.count),
        float(workflow.sink.nprocs),
        float(workflow.sink.sleep_seconds),
        float(workflow.sink.iterations),
    ]
    return dict(zip(FEATURE_NAMES, source + intermediate + sink))


def _role_mean(tasks: Sequence[Mapping[str, Any]], role: str, field: str) -> float:
    values = [float(task[field]) for task in tasks if task.get("role") == role]
    return mean(values) if values else 0.0


def extract_targets(summary: Mapping[str, Any]) -> Dict[str, float]:
    """Extract the eight measured outputs used by the performance model."""

    tasks = summary.get("tasks")
    if isinstance(tasks, (str, bytes)) or not isinstance(tasks, Sequence):
        raise ValueError("summary.tasks must be a sequence")

    values = (
        _role_mean(tasks, "source", "task_time_ms"),
        _role_mean(tasks, "intermediate", "task_time_ms"),
        _role_mean(tasks, "sink", "task_time_ms"),
        _role_mean(tasks, "source", "output_time_ms"),
        _role_mean(tasks, "intermediate", "input_time_ms"),
        _role_mean(tasks, "intermediate", "output_time_ms"),
        _role_mean(tasks, "sink", "input_time_ms"),
        float(summary["e2e_time_seconds"]),
    )
    return dict(zip(TARGET_NAMES, values))


def observation_row(
    features: Mapping[str, float], targets: Mapping[str, float]
) -> List[float]:
    """Return one CSV row in the canonical order."""

    return [float(features[name]) for name in FEATURE_NAMES] + [
        float(targets[name]) for name in TARGET_NAMES
    ]


def append_observation(
    path: Path, features: Mapping[str, float], targets: Mapping[str, float]
) -> None:
    """Append one observation, rejecting files with an incompatible schema."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", newline="") as stream:
        try:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        except ImportError:  # pragma: no cover - WfPerf targets Unix HPC systems
            pass

        stream.seek(0)
        first_line = stream.readline()
        if first_line:
            header = next(csv.reader([first_line]))
            if tuple(header) != DATABASE_COLUMNS:
                raise ValueError(
                    "{} has an incompatible WfPerf ML schema".format(path)
                )
        else:
            csv.writer(stream).writerow(DATABASE_COLUMNS)

        stream.seek(0, os.SEEK_END)
        csv.writer(stream).writerow(observation_row(features, targets))
