# Copyright (c) 2026, University of Florida. All rights reserved.
#
# This file is part of WfPerf, developed by members of Prof. Xiaoyi Lu's
# group at the University of Florida. See LICENSE in the top-level directory.

from pathlib import Path

import pytest

from wfperf.config import ConfigError, load_workflow, parse_workflow


def test_load_benchmark_configuration():
    path = Path(__file__).parents[1] / "examples" / "benchmark.yaml"
    workflow = load_workflow(path)

    assert workflow.source.count == 1
    assert len(workflow.intermediates) == 2
    assert workflow.intermediates[0].particles_per_process == 1_000_000
    assert workflow.sink.count == 1
    assert workflow.task_count == 4


def test_rejects_wrong_intermediate_array_length():
    data = {
        "tasks": [
            {
                "func": "source",
                "count": 1,
                "nprocs": 1,
                "sleep": 0,
                "iter": 1,
                "particle": 1,
            },
            {
                "func": "intermediate",
                "number of intermediate": 2,
                "count": [1],
                "nprocs": [1, 1],
                "sleep": [0, 0],
                "iter": [1, 1],
                "particle": [1, 1],
            },
            {
                "func": "sink",
                "count": 1,
                "nprocs": 1,
                "sleep": 0,
                "iter": 1,
            },
        ]
    }

    with pytest.raises(ConfigError, match="must contain 2 values"):
        parse_workflow(data)


def test_accepts_workflow_without_intermediate_stages():
    data = {
        "tasks": [
            {
                "func": "source",
                "count": 1,
                "nprocs": 1,
                "sleep": 0,
                "iter": 1,
                "particle": 1,
            },
            {"func": "intermediate", "number of intermediate": 0},
            {
                "func": "sink",
                "count": 1,
                "nprocs": 1,
                "sleep": 0,
                "iter": 1,
            },
        ]
    }

    assert parse_workflow(data).intermediates == ()
