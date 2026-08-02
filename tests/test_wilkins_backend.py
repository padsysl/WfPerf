# Copyright (c) 2026, University of Florida. All rights reserved.
#
# This file is part of WfPerf, developed by members of Prof. Xiaoyi Lu's
# group at the University of Florida. See LICENSE in the top-level directory.

from types import SimpleNamespace

import pytest
import yaml

from wfperf.backends.wilkins.backend import (
    WilkinsBackend,
    parse_wilkins_metrics,
    render_wilkins_workflow,
)
from wfperf.config import StageSpec, WorkflowSpec


def workflow():
    return WorkflowSpec(
        source=StageSpec("source", 0, 1, 2, 0, 1, 8),
        intermediates=(StageSpec("intermediate", 0, 1, 1, 0, 1, 8),),
        sink=StageSpec("sink", 0, 1, 1, 0, 1),
    )


def wilkins_output():
    return """
[0] Producer Task Time: 10 ms
[1] Producer Task Time: 14 ms
[0] Producer H5Fcreate to H5Fclose Time: 2 ms
[1] Producer H5Fcreate to H5Fclose Time: 4 ms
[2] Middle0 Task Time: 20 ms
[2] Middle0 Input H5Fcreate to H5Fclose Time: 5 ms
[2] Middle0 Output H5Fcreate to H5Fclose Time: 6 ms
[3] Consumer Task Time: 30 ms
[3] Consumer H5Fcreate to H5Fclose Time: 7 ms
"""


def test_render_wilkins_workflow_uses_common_specification():
    decoded = yaml.safe_load(render_wilkins_workflow(workflow()))

    assert [task["func"] for task in decoded["tasks"]] == [
        "prod-henson0",
        "middle0",
        "con-henson0",
    ]
    assert decoded["tasks"][0]["nprocs"] == 2
    assert decoded["tasks"][1]["args"][4] == "8"


def test_parse_wilkins_metrics_uses_common_result_schema():
    tasks = parse_wilkins_metrics(workflow(), wilkins_output())

    assert len(tasks) == 3
    assert tasks[0]["backend"] == "wilkins"
    assert tasks[0]["task_time_ms"] == 12
    assert tasks[0]["output_time_ms"] == 3
    assert tasks[1]["input_time_ms"] == 5
    assert tasks[2]["input_time_ms"] == 7


def test_wilkins_backend_stages_runtime_and_writes_summary(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    for name in ("prod-henson.hx", "middle.hx", "con-henson.hx"):
        (runtime / name).write_text("module")
    installed = tmp_path / "installed"
    installed.mkdir()
    master = installed / "wilkins-master.py"
    master.write_text("# installed Wilkins driver\n")

    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["cwd"] = kwargs["cwd"]
        return SimpleNamespace(returncode=0, stdout=wilkins_output())

    monkeypatch.setattr("wfperf.backends.wilkins.backend.subprocess.run", fake_run)
    result = WilkinsBackend(runtime, driver=str(master)).run(
        workflow(), tmp_path / "runs"
    )

    assert result.summary["backend"] == "wilkins"
    assert len(result.summary["tasks"]) == 3
    assert result.summary_path.is_file()
    assert captured["command"][:4] == ["mpirun", "-l", "-n", "4"]
    assert captured["command"][6] == str(master)
    assert result.summary["wilkins_driver"] == str(master)
    assert not (result.run_directory / "wilkins-master.py").exists()


def test_wilkins_backend_requires_an_installed_driver(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    for name in ("prod-henson.hx", "middle.hx", "con-henson.hx"):
        (runtime / name).write_text("module")

    backend = WilkinsBackend(runtime, driver="definitely-not-a-wilkins-driver")
    with pytest.raises(FileNotFoundError, match="installed Wilkins driver"):
        backend.run(workflow(), tmp_path / "runs")
