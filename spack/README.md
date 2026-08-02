# Spack support

This directory contains a custom Spack 1.x package repository for WfPerf:

```text
spack/
├── spack-repo-index.yaml
└── spack_repo/wfperf/
    ├── repo.yaml
    └── packages/wfperf/package.py
```

The `wfperf` package has two independent backend variants and one optional
performance-model variant:

| Variant | Dependencies |
| --- | --- |
| `+parsl` | Python 3.10+, Parsl, h5py, and NumPy |
| `+wilkins` | MPI, parallel HDF5, mpi4py, Henson, LowFive, CMake, and Wilkins |
| `+ml` | scikit-learn, NumPy, and joblib |

Neither backend is selected implicitly. Users can install either backend or
both, and select the execution backend for each WfPerf run.

## Install from a source checkout

From the WfPerf repository root, register the package repository and create an
environment:

```bash
spack repo add "$PWD/spack/spack_repo/wfperf"
spack env create wfperf
spack env activate wfperf
spack develop --path "$PWD" wfperf@main
```

Choose one of these specs:

```bash
# Parsl only
spack add wfperf@main+parsl~wilkins

# Wilkins only
spack add wfperf@main~parsl+wilkins

# Both backends
spack add wfperf@main+parsl+wilkins

# Add model training and inference to either selection
spack add wfperf@main+parsl~wilkins+ml
```

Then concretize and install:

```bash
spack concretize
spack install
spack load wfperf
```

Run with the selected backend:

```bash
wfperf examples/benchmark.yaml --backend parsl

wfperf examples/benchmark.yaml --backend wilkins
```

For `+wilkins`, the package compiles WfPerf's native task modules into
`$WFPERF_WILKINS_RUNTIME`. The environment variable is set by
`spack load wfperf`; it may also be overridden with `--wilkins-runtime`.

Use `spack spec -I wfperf@main+parsl` or
`spack spec -I wfperf@main+wilkins` before installation to inspect the selected
dependency graph.

## Wilkins package repository

The Wilkins and LowFive package recipes are maintained with those external
projects rather than duplicated in WfPerf. Their package repository must be
registered with Spack before concretizing `wfperf+wilkins`. This keeps WfPerf
focused on the benchmark and avoids vendoring dependency source trees. WfPerf
does not redistribute those projects; users install them separately.

## Compatibility and site configuration

The repository uses Spack's version 2 package-repository API and requires
Spack 1.x. No cluster allocation, scheduler, compiler, or concretized
environment is encoded here. Sites select those settings in their own Spack
configuration or environment.
