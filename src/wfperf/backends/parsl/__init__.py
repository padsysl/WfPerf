# Copyright (c) 2026, University of Florida. All rights reserved.
#
# This file is part of WfPerf, developed by members of Prof. Xiaoyi Lu's
# group at the University of Florida. See LICENSE in the top-level directory.

"""Parsl backend for distributed WfPerf workflows."""

from wfperf.backends.parsl.backend import ParslBackend, RunResult

__all__ = ["ParslBackend", "RunResult"]
