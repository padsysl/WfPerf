# Wilkins backend

This backend maps the common WfPerf source/intermediate/sink schema to an
externally installed Wilkins runtime. WfPerf supplies its own MPI/HDF5 task
programs; Wilkins, Henson, and LowFive remain external dependencies. The
backend emits the same `summary.json` fields as the Parsl backend.

## Usage

Build the WfPerf native modules, place `prod-henson.hx`, `middle.hx`, and
`con-henson.hx` in one directory, load the Wilkins environment so that its
`wilkins-master.py` driver is in `PATH`, and run:

```bash
wfperf examples/benchmark.yaml --backend wilkins \
    --wilkins-runtime /path/to/wfperf/modules
```

The default launch command is `mpirun -l`. It can be changed without editing
WfPerf:

```bash
wfperf examples/benchmark.yaml --backend wilkins \
    --wilkins-runtime /path/to/wfperf/modules \
    --wilkins-launcher mpiexec \
    --wilkins-launcher-args="-l"
```

`--wilkins-driver /path/to/wilkins-master.py` may be used when the installed
driver is not available in `PATH`. The driver is never copied into or
distributed with WfPerf.

The backend creates an isolated run directory, translates the shared YAML to
the Wilkins schema, stages per-task module names, invokes the installed driver,
captures `wilkins.log`, and aggregates rank-labelled timings into logical
WfPerf tasks.
