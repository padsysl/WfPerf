# Copyright (c) 2026, University of Florida. All rights reserved.
#
# This file is part of WfPerf, developed by members of Prof. Xiaoyi Lu's
# group at the University of Florida. See LICENSE in the top-level directory.

import json

import pytest

pytest.importorskip("h5py")
pytest.importorskip("parsl")

from wfperf.backends.parsl.backend import ParslBackend
from wfperf.config import StageSpec, WorkflowSpec


def test_local_backend_executes_complete_graph(tmp_path):
    workflow = WorkflowSpec(
        source=StageSpec("source", 0, 2, 1, 0, 1, 8),
        intermediates=(StageSpec("intermediate", 0, 1, 1, 0, 1, 8),),
        sink=StageSpec("sink", 0, 2, 1, 0, 1),
    )

    result = ParslBackend(max_workers=4).run(workflow, tmp_path)

    assert result.summary["backend"] == "parsl"
    assert len(result.summary["tasks"]) == 5
    assert json.loads(result.summary_path.read_text())["schema_version"] == 1
    assert not (result.run_directory / "data").exists()
