# Copyright (c) 2026, University of Florida. All rights reserved.
#
# This file is part of WfPerf, developed by members of Prof. Xiaoyi Lu's
# group at the University of Florida. See LICENSE in the top-level directory.

"""HDF5 workflow blocks executed by Parsl workers."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


DATASET = "/group1/particles"
DIMENSIONS = 3
CHUNK_ROWS = 65536


def _write_particles(path: str, particles_per_process: int, nprocs: int, value: float) -> None:
    import h5py
    import numpy as np

    rows = particles_per_process * nprocs
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    chunk_rows = min(CHUNK_ROWS, rows)
    with h5py.File(str(output), "w") as handle:
        dataset = handle.create_dataset(
            DATASET,
            shape=(rows, DIMENSIONS),
            dtype="f4",
            chunks=(chunk_rows, DIMENSIONS),
        )
        block = np.full((chunk_rows, DIMENSIONS), value, dtype=np.float32)
        for start in range(0, rows, chunk_rows):
            stop = min(start + chunk_rows, rows)
            dataset[start:stop, :] = block[: stop - start, :]


def _read_particles(paths: Sequence[str]) -> float:
    import h5py

    checksum = 0.0
    for path in paths:
        with h5py.File(path, "r") as handle:
            dataset = handle[DATASET]
            rows = dataset.shape[0]
            for start in range(0, rows, CHUNK_ROWS):
                block = dataset[start : min(start + CHUNK_ROWS, rows), :]
                if block.size:
                    checksum += float(block[0, 0])
    return checksum


def _metrics(
    role: str,
    stage: int,
    task_index: int,
    nprocs: int,
    iterations: int,
    task_time_ms: float,
    input_time_ms: float = 0.0,
    output_time_ms: float = 0.0,
    particles_per_process: Optional[int] = None,
    input_files: Sequence[str] = (),
    output_file: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "backend": "parsl",
        "role": role,
        "stage": stage,
        "task_index": task_index,
        "nprocs": nprocs,
        "iterations": iterations,
        "particles_per_process": particles_per_process,
        "task_time_ms": task_time_ms,
        "input_time_ms": input_time_ms,
        "output_time_ms": output_time_ms,
        "input_files": list(input_files),
        "output_file": output_file,
    }


def run_producer(
    task_index: int,
    nprocs: int,
    iterations: int,
    sleep_seconds: float,
    particles_per_process: int,
    output_file: str,
) -> Dict[str, Any]:
    """Generate an HDF5 dataset and return Wilkins-equivalent timers."""

    task_start = time.perf_counter()
    output_time = 0.0
    for iteration in range(iterations):
        time.sleep(sleep_seconds)
        output_start = time.perf_counter()
        _write_particles(output_file, particles_per_process, nprocs, task_index + iteration)
        output_time += time.perf_counter() - output_start

    return _metrics(
        role="source",
        stage=0,
        task_index=task_index,
        nprocs=nprocs,
        iterations=iterations,
        particles_per_process=particles_per_process,
        task_time_ms=(time.perf_counter() - task_start) * 1000.0,
        output_time_ms=(output_time / iterations) * 1000.0,
        output_file=output_file,
    )


def run_intermediate(
    stage: int,
    task_index: int,
    nprocs: int,
    iterations: int,
    sleep_seconds: float,
    particles_per_process: int,
    input_files: Sequence[str],
    output_file: str,
) -> Dict[str, Any]:
    """Consume upstream HDF5 data, emulate compute, and produce new HDF5 data."""

    task_start = time.perf_counter()
    input_time = 0.0
    input_operations = 0
    output_time = 0.0
    for iteration in range(iterations):
        checksum = 0.0
        for input_file in input_files:
            input_start = time.perf_counter()
            checksum += _read_particles([input_file])
            input_time += time.perf_counter() - input_start
            input_operations += 1
            time.sleep(sleep_seconds)
        output_start = time.perf_counter()
        _write_particles(
            output_file,
            particles_per_process,
            nprocs,
            checksum + task_index + iteration,
        )
        output_time += time.perf_counter() - output_start

    return _metrics(
        role="intermediate",
        stage=stage,
        task_index=task_index,
        nprocs=nprocs,
        iterations=iterations,
        particles_per_process=particles_per_process,
        task_time_ms=(time.perf_counter() - task_start) * 1000.0,
        input_time_ms=(input_time / input_operations) * 1000.0,
        output_time_ms=(output_time / iterations) * 1000.0,
        input_files=input_files,
        output_file=output_file,
    )


def run_sink(
    task_index: int,
    nprocs: int,
    iterations: int,
    sleep_seconds: float,
    input_files: Sequence[str],
) -> Dict[str, Any]:
    """Consume the final HDF5 datasets and return input/task timers."""

    task_start = time.perf_counter()
    input_time = 0.0
    input_operations = 0
    for _ in range(iterations):
        for input_file in input_files:
            input_start = time.perf_counter()
            _read_particles([input_file])
            input_time += time.perf_counter() - input_start
            input_operations += 1
            time.sleep(sleep_seconds)

    return _metrics(
        role="sink",
        stage=0,
        task_index=task_index,
        nprocs=nprocs,
        iterations=iterations,
        task_time_ms=(time.perf_counter() - task_start) * 1000.0,
        input_time_ms=(input_time / input_operations) * 1000.0,
        input_files=input_files,
    )
