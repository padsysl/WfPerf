# Copyright (c) 2026, University of Florida. All rights reserved.
#
# This file is part of WfPerf, developed by members of Prof. Xiaoyi Lu's
# group at the University of Florida. See LICENSE in the top-level directory.

"""Backend adapter for executing WfPerf workflows with Wilkins."""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence, Tuple

import yaml

from wfperf.config import StageSpec, WorkflowSpec
from wfperf.result import RunResult, average_metrics


def _outport(stage: int) -> Dict[str, Any]:
    return {
        "filename": "outfile{}*.h5".format(stage),
        "dsets": [{"name": "/group1/particles", "passthru": 0, "metadata": 1}],
    }


def _inport(stage: int, passthru: int, metadata: int) -> Dict[str, Any]:
    result = {
        "filename": "outfile{}*.h5".format(stage),
        "dsets": [
            {
                "name": "/group1/particles",
                "passthru": passthru,
                "metadata": metadata,
            }
        ],
    }
    if passthru:
        result["io_freq"] = 1
    return result


def render_wilkins_workflow(workflow: WorkflowSpec) -> str:
    """Translate the common WfPerf model to the Wilkins YAML schema."""

    source = workflow.source
    tasks: List[Dict[str, Any]] = [
        {
            "taskCount": source.count,
            "func": "prod-henson0",
            "nprocs": source.nprocs,
            "args": [
                str(source.iterations),
                str(source.sleep_seconds),
                str(source.particles_per_process),
                str(workflow.sink.count),
            ],
            "outports": [_outport(0)],
        }
    ]

    upstream_count = source.count
    for stage in workflow.intermediates:
        inputs_per_task = int(math.ceil(float(upstream_count) / stage.count))
        tasks.append(
            {
                "taskCount": stage.count,
                "func": "middle{}".format(stage.index),
                "nprocs": stage.nprocs,
                "args": [
                    str(stage.iterations),
                    str(stage.sleep_seconds),
                    str(stage.index),
                    str(stage.index + 1),
                    str(stage.particles_per_process),
                    str(inputs_per_task),
                    "{filename}",
                ],
                "inports": [_inport(stage.index, passthru=0, metadata=1)],
                "outports": [_outport(stage.index + 1)],
            }
        )
        upstream_count = stage.count

    final_stage = len(workflow.intermediates)
    inputs_per_sink = int(math.ceil(float(upstream_count) / workflow.sink.count))
    sink = workflow.sink
    tasks.append(
        {
            "taskCount": sink.count,
            "func": "con-henson0",
            "nprocs": sink.nprocs,
            "args": [
                str(sink.iterations),
                str(inputs_per_sink),
                "{filename}",
                str(final_stage),
                str(sink.sleep_seconds),
            ],
            "inports": [_inport(final_stage, passthru=1, metadata=0)],
        }
    )
    return yaml.safe_dump({"tasks": tasks}, sort_keys=False)


def _runtime_file(runtime: Path, names: Sequence[str]) -> Path:
    for name in names:
        candidate = runtime / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "WfPerf module directory {} is missing {}".format(
            runtime, " or ".join(names)
        )
    )


def _installed_driver(command: str) -> Path:
    supplied = Path(command).expanduser()
    if supplied.is_absolute() or supplied.parent != Path("."):
        candidate = supplied.resolve()
        if candidate.is_file():
            return candidate
    else:
        discovered = shutil.which(command)
        if discovered:
            return Path(discovered).resolve()
    raise FileNotFoundError(
        "cannot find the installed Wilkins driver {!r}; load the Wilkins "
        "environment or pass --wilkins-driver".format(command)
    )


def _copy_modules(workflow: WorkflowSpec, runtime: Path, run_directory: Path) -> None:
    producer = _runtime_file(runtime, ("prod-henson.hx", "prod-henson0.hx"))
    middle = None
    if workflow.intermediates:
        middle = _runtime_file(runtime, ("middle.hx",))
    consumer = _runtime_file(runtime, ("con-henson.hx", "con-henson0.hx"))

    shutil.copy2(str(producer), str(run_directory / "prod-henson0.hx"))
    for index in range(workflow.source.count):
        shutil.copy2(str(producer), str(run_directory / "prod-henson0_{}.hx".format(index)))

    for stage in workflow.intermediates:
        assert middle is not None
        if stage.count == 1:
            shutil.copy2(str(middle), str(run_directory / "middle{}.hx".format(stage.index)))
        else:
            for index in range(stage.count):
                name = "middle{}_{}.hx".format(stage.index, index)
                shutil.copy2(str(middle), str(run_directory / name))

    shutil.copy2(str(consumer), str(run_directory / "con-henson0.hx"))
    for index in range(workflow.sink.count):
        shutil.copy2(str(consumer), str(run_directory / "con-henson0_{}.hx".format(index)))


def _rank_map(workflow: WorkflowSpec) -> Dict[int, Tuple[StageSpec, int]]:
    result = {}
    rank = 0
    for stage in workflow.stages:
        for task_index in range(stage.count):
            for _ in range(stage.nprocs):
                result[rank] = (stage, task_index)
                rank += 1
    return result


_TIMING_PATTERN = re.compile(
    r"\[(?P<rank>\d+)\]\s+"
    r"(?P<label>Producer|Consumer|Middle(?P<middle>\d+))\s+"
    r"(?P<metric>Task|Input H5Fcreate to H5Fclose|"
    r"Output H5Fcreate to H5Fclose|H5Fcreate to H5Fclose)\s+Time:\s+"
    r"(?P<value>[\d.]+)\s+ms"
)


def parse_wilkins_metrics(workflow: WorkflowSpec, output: str) -> List[Dict[str, Any]]:
    """Convert rank-labelled Wilkins timers to the common logical-task schema."""

    ranks = _rank_map(workflow)
    observed: Dict[Tuple[str, int, int], Dict[str, List[float]]] = {}
    for match in _TIMING_PATTERN.finditer(output):
        rank = int(match.group("rank"))
        if rank not in ranks:
            continue
        stage, task_index = ranks[rank]
        label = match.group("label")
        expected = (
            "Producer"
            if stage.role == "source"
            else "Consumer"
            if stage.role == "sink"
            else "Middle{}".format(stage.index)
        )
        if label != expected:
            continue

        metric_label = match.group("metric")
        metric = "task_time_ms"
        if metric_label.startswith("Input") or (
            stage.role == "sink" and metric_label.startswith("H5Fcreate")
        ):
            metric = "input_time_ms"
        elif metric_label.startswith("Output") or (
            stage.role == "source" and metric_label.startswith("H5Fcreate")
        ):
            metric = "output_time_ms"

        key = (stage.role, stage.index, task_index)
        values = observed.setdefault(
            key,
            {"task_time_ms": [], "input_time_ms": [], "output_time_ms": []},
        )
        values[metric].append(float(match.group("value")))

    if not observed:
        raise ValueError("Wilkins output contained no recognized rank-labelled timers")

    tasks = []
    for stage in workflow.stages:
        for task_index in range(stage.count):
            values = observed.get(
                (stage.role, stage.index, task_index),
                {"task_time_ms": [], "input_time_ms": [], "output_time_ms": []},
            )
            tasks.append(
                {
                    "backend": "wilkins",
                    "role": stage.role,
                    "stage": stage.index,
                    "task_index": task_index,
                    "nprocs": stage.nprocs,
                    "iterations": stage.iterations,
                    "particles_per_process": stage.particles_per_process,
                    "task_time_ms": mean(values["task_time_ms"])
                    if values["task_time_ms"]
                    else 0.0,
                    "input_time_ms": mean(values["input_time_ms"])
                    if values["input_time_ms"]
                    else 0.0,
                    "output_time_ms": mean(values["output_time_ms"])
                    if values["output_time_ms"]
                    else 0.0,
                    "input_files": [],
                    "output_file": None,
                }
            )
    return tasks


class WilkinsBackend:
    """Execute a WfPerf workflow through an externally installed Wilkins driver."""

    def __init__(
        self,
        runtime_directory: Any,
        launcher: str = "mpirun",
        launcher_args: Sequence[str] = ("-l",),
        driver: str = "wilkins-master.py",
    ):
        self.runtime_directory = Path(runtime_directory).resolve()
        self.launcher = launcher
        self.launcher_args = tuple(launcher_args)
        self.driver = driver

    def run(
        self,
        workflow: WorkflowSpec,
        output_directory: Any,
        keep_files: bool = False,
    ) -> RunResult:
        output_root = Path(output_directory).resolve()
        run_id = "{}-{}".format(
            datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"), uuid.uuid4().hex[:8]
        )
        run_directory = output_root / run_id
        run_directory.mkdir(parents=True, exist_ok=False)

        _copy_modules(workflow, self.runtime_directory, run_directory)
        driver = _installed_driver(self.driver)
        workflow_path = run_directory / "wilkins-workflow.yaml"
        workflow_path.write_text(render_wilkins_workflow(workflow))

        rank_count = sum(stage.count * stage.nprocs for stage in workflow.stages)
        command = [self.launcher]
        command.extend(self.launcher_args)
        command.extend(
            [
                "-n",
                str(rank_count),
                sys.executable,
                "-u",
                str(driver),
                workflow_path.name,
                "-s",
            ]
        )

        started_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        start = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=str(run_directory),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        e2e_time = time.perf_counter() - start
        log_path = run_directory / "wilkins.log"
        log_path.write_text(completed.stdout)
        if completed.returncode:
            raise RuntimeError(
                "Wilkins exited with status {}; see {}".format(
                    completed.returncode, log_path
                )
            )

        task_metrics = parse_wilkins_metrics(workflow, completed.stdout)
        summary = {
            "schema_version": 1,
            "backend": "wilkins",
            "run_id": run_id,
            "started_at": started_at,
            "e2e_time_seconds": e2e_time,
            "tasks": task_metrics,
            "averages": average_metrics(task_metrics),
            "launcher": command,
            "wilkins_driver": str(driver),
        }
        summary_path = run_directory / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

        if not keep_files:
            for path in run_directory.glob("*.h5"):
                path.unlink()

        return RunResult(run_directory, summary_path, summary)
