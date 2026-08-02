# Copyright (c) 2026, University of Florida. All rights reserved.
#
# This file is part of WfPerf, developed by members of Prof. Xiaoyi Lu's
# group at the University of Florida. See LICENSE in the top-level directory.

"""Backend-neutral WfPerf result types and aggregation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence


@dataclass(frozen=True)
class RunResult:
    """Locations and metrics produced by one WfPerf invocation."""

    run_directory: Path
    summary_path: Path
    summary: Dict[str, Any]


def average_metrics(tasks: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Average common timing fields by source, intermediate stage, and sink."""

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for task in tasks:
        key = task["role"]
        if key == "intermediate":
            key = "intermediate{}".format(task["stage"])
        grouped.setdefault(key, []).append(task)

    result = {}
    for key, members in grouped.items():
        result[key] = {
            "task_time_ms": mean(item["task_time_ms"] for item in members),
            "input_time_ms": mean(item["input_time_ms"] for item in members),
            "output_time_ms": mean(item["output_time_ms"] for item in members),
        }
    return result
