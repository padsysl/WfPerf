# Copyright (c) 2026, University of Florida. All rights reserved.
#
# This file is part of WfPerf, developed by members of Prof. Xiaoyi Lu's
# group at the University of Florida. See LICENSE in the top-level directory.

import pytest

from wfperf.backends.parsl.backend import assign_inputs


def test_assign_inputs_fan_in():
    assert assign_inputs([0, 1, 2, 3], 2) == [[0, 1], [2, 3]]


def test_assign_inputs_fan_out():
    assert assign_inputs([0, 1], 5) == [[0], [1], [0], [1], [0]]


def test_assign_inputs_rejects_empty_upstream():
    with pytest.raises(ValueError, match="upstream"):
        assign_inputs([], 1)
