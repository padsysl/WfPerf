# Copyright (c) 2026, University of Florida. All rights reserved.
#
# This file is part of WfPerf, developed by members of Prof. Xiaoyi Lu's
# group at the University of Florida. See LICENSE in the top-level directory.

"""Backend-independent WfPerf configuration parsing and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import yaml


class ConfigError(ValueError):
    """Raised when a WfPerf configuration is incomplete or inconsistent."""


@dataclass(frozen=True)
class StageSpec:
    """Configuration shared by every task instance in one workflow stage."""

    role: str
    index: int
    count: int
    nprocs: int
    sleep_seconds: float
    iterations: int
    particles_per_process: Optional[int] = None

    @property
    def name(self) -> str:
        if self.role == "intermediate":
            return "intermediate{}".format(self.index)
        return self.role


@dataclass(frozen=True)
class WorkflowSpec:
    """Validated source/intermediate/sink workflow configuration."""

    source: StageSpec
    intermediates: Tuple[StageSpec, ...]
    sink: StageSpec

    @property
    def stages(self) -> Tuple[StageSpec, ...]:
        return (self.source,) + self.intermediates + (self.sink,)

    @property
    def task_count(self) -> int:
        return sum(stage.count for stage in self.stages)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError("{} must be a mapping".format(label))
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError("{} must be a positive integer".format(label))
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigError("{} must be a non-negative integer".format(label))
    return value


def _nonnegative_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ConfigError("{} must be a non-negative number".format(label))
    return float(value)


def _required(task: Mapping[str, Any], key: str, label: str) -> Any:
    if key not in task:
        raise ConfigError("{}.{} is required".format(label, key))
    return task[key]


def _sequence(task: Mapping[str, Any], key: str, length: int, label: str) -> Sequence[Any]:
    value = _required(task, key, label)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigError("{}.{} must be a list".format(label, key))
    if len(value) != length:
        raise ConfigError(
            "{}.{} must contain {} values, found {}".format(label, key, length, len(value))
        )
    return value


def _single_stage(task: Mapping[str, Any], role: str, particles: bool) -> StageSpec:
    label = "task '{}'".format(role)
    return StageSpec(
        role=role,
        index=0,
        count=_positive_int(_required(task, "count", label), "{}.count".format(label)),
        nprocs=_positive_int(_required(task, "nprocs", label), "{}.nprocs".format(label)),
        sleep_seconds=_nonnegative_float(
            _required(task, "sleep", label), "{}.sleep".format(label)
        ),
        iterations=_positive_int(_required(task, "iter", label), "{}.iter".format(label)),
        particles_per_process=(
            _positive_int(_required(task, "particle", label), "{}.particle".format(label))
            if particles
            else None
        ),
    )


def parse_workflow(data: Mapping[str, Any]) -> WorkflowSpec:
    """Validate a decoded YAML mapping and return a backend-neutral workflow."""

    root = _mapping(data, "configuration")
    tasks = root.get("tasks")
    if isinstance(tasks, (str, bytes)) or not isinstance(tasks, Sequence):
        raise ConfigError("configuration.tasks must be a list")

    by_role: Dict[str, Mapping[str, Any]] = {}
    for position, value in enumerate(tasks):
        task = _mapping(value, "tasks[{}]".format(position))
        role = task.get("func")
        if role not in ("source", "intermediate", "sink"):
            raise ConfigError("tasks[{}].func has unsupported value {!r}".format(position, role))
        if role in by_role:
            raise ConfigError("configuration contains more than one '{}' entry".format(role))
        by_role[role] = task

    if "source" not in by_role or "sink" not in by_role:
        raise ConfigError("configuration requires exactly one source and one sink entry")

    source = _single_stage(by_role["source"], "source", particles=True)
    sink = _single_stage(by_role["sink"], "sink", particles=False)

    intermediates = []
    intermediate = by_role.get("intermediate")
    if intermediate is not None:
        label = "task 'intermediate'"
        count = _nonnegative_int(
            _required(intermediate, "number of intermediate", label),
            "{}.number of intermediate".format(label),
        )
        if count:
            counts = _sequence(intermediate, "count", count, label)
            nprocs = _sequence(intermediate, "nprocs", count, label)
            sleeps = _sequence(intermediate, "sleep", count, label)
            iterations = _sequence(intermediate, "iter", count, label)
            particles = _sequence(intermediate, "particle", count, label)
            for index in range(count):
                item_label = "intermediate[{}]".format(index)
                intermediates.append(
                    StageSpec(
                        role="intermediate",
                        index=index,
                        count=_positive_int(counts[index], "{}.count".format(item_label)),
                        nprocs=_positive_int(nprocs[index], "{}.nprocs".format(item_label)),
                        sleep_seconds=_nonnegative_float(
                            sleeps[index], "{}.sleep".format(item_label)
                        ),
                        iterations=_positive_int(
                            iterations[index], "{}.iter".format(item_label)
                        ),
                        particles_per_process=_positive_int(
                            particles[index], "{}.particle".format(item_label)
                        ),
                    )
                )

    return WorkflowSpec(source=source, intermediates=tuple(intermediates), sink=sink)


def load_workflow(path: Any) -> WorkflowSpec:
    """Load and validate a WfPerf YAML file."""

    config_path = Path(path)
    try:
        decoded = yaml.safe_load(config_path.read_text())
    except OSError as error:
        raise ConfigError("cannot read {}: {}".format(config_path, error)) from error
    except yaml.YAMLError as error:
        raise ConfigError("invalid YAML in {}: {}".format(config_path, error)) from error
    return parse_workflow(decoded)
