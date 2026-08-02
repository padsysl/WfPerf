# Copyright (c) 2026, University of Florida. All rights reserved.
#
# This file is part of WfPerf, developed by members of Prof. Xiaoyi Lu's
# group at the University of Florida. See LICENSE in the top-level directory.

"""Workflow construction, execution, and reporting for the Parsl backend."""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Sequence

from wfperf.config import WorkflowSpec
from wfperf.result import RunResult, average_metrics


def assign_inputs(upstream: Sequence[Any], downstream_count: int) -> List[List[Any]]:
    """Balance upstream outputs over downstream tasks for fan-in and fan-out."""

    if not upstream:
        raise ValueError("a downstream stage requires at least one upstream output")
    if downstream_count < 1:
        raise ValueError("downstream_count must be positive")

    upstream_count = len(upstream)
    if downstream_count <= upstream_count:
        groups = []
        for index in range(downstream_count):
            start = (index * upstream_count) // downstream_count
            stop = ((index + 1) * upstream_count) // downstream_count
            groups.append(list(upstream[start:stop]))
        return groups

    return [[upstream[index % upstream_count]] for index in range(downstream_count)]


def thread_pool_config(max_workers: int, run_dir: Path):
    """Create a provider-free Parsl thread-pool configuration."""

    from parsl.config import Config
    from parsl.executors import ThreadPoolExecutor

    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    return Config(
        executors=[
            ThreadPoolExecutor(label="wfperf-thread-pool", max_threads=max_workers)
        ],
        run_dir=str(run_dir),
        usage_tracking=False,
    )


class ParslBackend:
    """Execute validated WfPerf workflows using Parsl applications."""

    def __init__(self, config: Any = None, max_workers: Optional[int] = None):
        self.config = config
        self.max_workers = max_workers

    def run(
        self,
        workflow: WorkflowSpec,
        output_directory: Any,
        keep_files: bool = False,
    ) -> RunResult:
        import parsl
        from parsl import File

        from wfperf.backends.parsl.apps import intermediate_app, producer_app, sink_app

        output_root = Path(output_directory).resolve()
        run_id = "{}-{}".format(
            datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"), uuid.uuid4().hex[:8]
        )
        run_directory = output_root / run_id
        data_directory = run_directory / "data"
        data_directory.mkdir(parents=True, exist_ok=False)

        max_workers = self.max_workers or max(1, min(workflow.task_count, 32))
        config = self.config or thread_pool_config(max_workers, run_directory / "parsl")
        generated_files: List[Path] = []
        metric_futures = []
        started_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        start = time.perf_counter()

        dfk = parsl.load(config)
        try:
            upstream = []
            source = workflow.source
            for task_index in range(source.count):
                output = data_directory / "source-{}.h5".format(task_index)
                generated_files.append(output)
                future = producer_app(
                    task_index,
                    source.nprocs,
                    source.iterations,
                    source.sleep_seconds,
                    source.particles_per_process,
                    outputs=[File(str(output))],
                )
                metric_futures.append(future)
                upstream.append(future.outputs[0])

            for stage in workflow.intermediates:
                dependencies = assign_inputs(upstream, stage.count)
                current = []
                for task_index, inputs in enumerate(dependencies):
                    output = data_directory / "intermediate{}-{}.h5".format(
                        stage.index, task_index
                    )
                    generated_files.append(output)
                    future = intermediate_app(
                        stage.index,
                        task_index,
                        stage.nprocs,
                        stage.iterations,
                        stage.sleep_seconds,
                        stage.particles_per_process,
                        inputs=inputs,
                        outputs=[File(str(output))],
                    )
                    metric_futures.append(future)
                    current.append(future.outputs[0])
                upstream = current

            sink_dependencies = assign_inputs(upstream, workflow.sink.count)
            for task_index, inputs in enumerate(sink_dependencies):
                future = sink_app(
                    task_index,
                    workflow.sink.nprocs,
                    workflow.sink.iterations,
                    workflow.sink.sleep_seconds,
                    inputs=inputs,
                )
                metric_futures.append(future)

            task_metrics = [future.result() for future in metric_futures]
            e2e_time = time.perf_counter() - start
        finally:
            dfk.cleanup()
            parsl.clear()

        summary = {
            "schema_version": 1,
            "backend": "parsl",
            "run_id": run_id,
            "started_at": started_at,
            "e2e_time_seconds": e2e_time,
            "tasks": task_metrics,
            "averages": average_metrics(task_metrics),
        }
        summary_path = run_directory / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

        if not keep_files:
            for path in generated_files:
                path.unlink(missing_ok=True)
            try:
                data_directory.rmdir()
            except OSError:
                pass

        return RunResult(
            run_directory=run_directory,
            summary_path=summary_path,
            summary=summary,
        )
