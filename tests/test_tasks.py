# Copyright (c) 2026, University of Florida. All rights reserved.
#
# This file is part of WfPerf, developed by members of Prof. Xiaoyi Lu's
# group at the University of Florida. See LICENSE in the top-level directory.

import pytest

h5py = pytest.importorskip("h5py")

from wfperf.backends.parsl.tasks import DATASET, run_intermediate, run_producer, run_sink


def test_hdf5_tasks_round_trip(tmp_path):
    source = tmp_path / "source.h5"
    middle = tmp_path / "middle.h5"

    produced = run_producer(0, 2, 1, 0, 8, str(source))
    with h5py.File(str(source), "r") as handle:
        assert handle[DATASET].shape == (16, 3)

    transformed = run_intermediate(0, 0, 1, 1, 0, 5, [str(source)], str(middle))
    consumed = run_sink(0, 1, 1, 0, [str(middle)])

    assert produced["output_time_ms"] >= 0
    assert transformed["input_time_ms"] >= 0
    assert transformed["output_time_ms"] >= 0
    assert consumed["input_time_ms"] >= 0
