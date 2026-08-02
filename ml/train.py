# Copyright (c) 2026, University of Florida. All rights reserved.
#
# This file is part of WfPerf, developed by members of Prof. Xiaoyi Lu's
# group at the University of Florida. See LICENSE in the top-level directory.

"""Repository-level entry point for explicit, offline WfPerf model training."""

from wfperf.ml.training import main


if __name__ == "__main__":
    raise SystemExit(main())
