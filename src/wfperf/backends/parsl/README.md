# Parsl backend

This backend maps the WfPerf source/intermediate/sink YAML schema to a Parsl
dataflow graph. Source tasks generate HDF5 particle arrays, intermediate tasks
read their assigned upstream files and produce new arrays, and sink tasks read
the final files. Parsl `File` futures carry the dependencies between stages.

## Usage

From the repository root:

```bash
python -m pip install -e '.[parsl]'
wfperf examples/benchmark.yaml --backend parsl
```

Useful options are:

```text
--output-dir DIR       Run and summary location (default: wfperf-runs)
--max-workers N        Workers in the built-in thread-pool configuration
--parsl-config FILE    Import a site-owned Parsl Config
--keep-files           Retain generated HDF5 files
--json                 Print the complete summary
```

The built-in thread-pool executor provides a ready-to-run configuration for
single-host execution. Distributed deployments can supply a Parsl
configuration defining either a `get_config()` function or a `CONFIG` object:

```python
from parsl.config import Config
from parsl.executors import ThreadPoolExecutor


def get_config():
    return Config(
        executors=[ThreadPoolExecutor(label="site", max_threads=8)],
        usage_tracking=False,
    )
```

Run it with:

```bash
wfperf examples/benchmark.yaml --backend parsl --parsl-config site_config.py
```

## Execution semantics

For fan-in, consecutive upstream outputs are divided as evenly as possible
among downstream tasks. For fan-out, upstream outputs are assigned to
downstream tasks round-robin. Multiple intermediate stages are chained in the
order listed by the YAML arrays.

The portable backend does not launch MPI ranks itself. `nprocs` preserves the
Wilkins workload size by making each output contain `particle * nprocs` rows.
Actual worker placement and resource requests belong to the user-supplied
Parsl configuration. This keeps scheduler, queue, account, and machine details
out of the open-source benchmark.

The JSON summary reports end-to-end time plus per-task and per-stage averages
for task, input, and output time. Input/output values average the corresponding
I/O operations, matching the existing Wilkins reporting style. Intermediate
and sink tasks apply the configured sleep once per consumed upstream file.
