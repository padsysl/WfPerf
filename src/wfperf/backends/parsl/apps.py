# Copyright (c) 2026, University of Florida. All rights reserved.
#
# This file is part of WfPerf, developed by members of Prof. Xiaoyi Lu's
# group at the University of Florida. See LICENSE in the top-level directory.

"""Serializable Parsl applications for WfPerf workflow blocks."""

from parsl import python_app


@python_app
def producer_app(
    task_index,
    nprocs,
    iterations,
    sleep_seconds,
    particles_per_process,
    outputs=(),
):
    from wfperf.backends.parsl.tasks import run_producer

    return run_producer(
        task_index=task_index,
        nprocs=nprocs,
        iterations=iterations,
        sleep_seconds=sleep_seconds,
        particles_per_process=particles_per_process,
        output_file=outputs[0].filepath,
    )


@python_app
def intermediate_app(
    stage,
    task_index,
    nprocs,
    iterations,
    sleep_seconds,
    particles_per_process,
    inputs=(),
    outputs=(),
):
    from wfperf.backends.parsl.tasks import run_intermediate

    return run_intermediate(
        stage=stage,
        task_index=task_index,
        nprocs=nprocs,
        iterations=iterations,
        sleep_seconds=sleep_seconds,
        particles_per_process=particles_per_process,
        input_files=[item.filepath for item in inputs],
        output_file=outputs[0].filepath,
    )


@python_app
def sink_app(task_index, nprocs, iterations, sleep_seconds, inputs=()):
    from wfperf.backends.parsl.tasks import run_sink

    return run_sink(
        task_index=task_index,
        nprocs=nprocs,
        iterations=iterations,
        sleep_seconds=sleep_seconds,
        input_files=[item.filepath for item in inputs],
    )
