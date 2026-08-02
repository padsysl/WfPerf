# Copyright (c) 2026, University of Florida. All rights reserved.
#
# This file is part of WfPerf, developed by members of Prof. Xiaoyi Lu's
# group at the University of Florida. See LICENSE in the top-level directory.

import pytest

from wfperf.cli import _create_backend, build_parser


def test_backend_selection_is_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["workflow.yaml"])


def test_wilkins_selection_requires_runtime_directory(monkeypatch):
    monkeypatch.delenv("WFPERF_WILKINS_RUNTIME", raising=False)
    args = build_parser().parse_args(["workflow.yaml", "--backend", "wilkins"])
    with pytest.raises(ValueError, match="--wilkins-runtime is required"):
        _create_backend(args)


def test_rejects_options_for_the_other_backend(tmp_path):
    args = build_parser().parse_args(
        [
            "workflow.yaml",
            "--backend",
            "parsl",
            "--wilkins-runtime",
            str(tmp_path),
        ]
    )
    with pytest.raises(ValueError, match="only valid"):
        _create_backend(args)
