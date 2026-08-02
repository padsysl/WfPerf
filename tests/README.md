<!--
Copyright (c) 2026, University of Florida. All rights reserved.

This file is part of WfPerf, developed by members of Prof. Xiaoyi Lu's
group at the University of Florida. See LICENSE in the top-level directory.
-->

# Tests

The suite covers YAML validation, topology assignment, backend selection,
Wilkins translation and timing extraction, HDF5 task behavior, a complete
Parsl dataflow graph, ML observation storage, legacy-data loading, offline
training, and threshold-based inference. Paper experiment sweeps are not
included. Run it from the repository root with:

```bash
python -m pip install -e '.[test]'
pytest
```
