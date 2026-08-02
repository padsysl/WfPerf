# Copyright (c) 2026, University of Florida. All rights reserved.
#
# This file is part of WfPerf, developed by members of Prof. Xiaoyi Lu's
# group at the University of Florida. See LICENSE in the top-level directory.

"""Wilkins backend for in situ WfPerf workflows."""

from wfperf.backends.wilkins.backend import WilkinsBackend
from wfperf.result import RunResult

__all__ = ["RunResult", "WilkinsBackend"]
