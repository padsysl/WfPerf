# WfPerf performance model

WfPerf separates model training from benchmark execution. Training is an
explicit offline operation; inference runs automatically after each successful
benchmark whenever a trained model is available.

## Install the ML dependencies

```bash
python -m pip install -e '.[ml]'
```

By default, WfPerf keeps the observation database and trained artifact in the
user data directory:

```text
~/.local/share/wfperf/runs.csv
~/.local/share/wfperf/model.joblib
```

If `XDG_DATA_HOME` is set, `$XDG_DATA_HOME/wfperf` is used instead. Set
`WFPERF_ML_DATABASE` and `WFPERF_ML_MODEL`, or use the corresponding command
options, to select other locations.

## Collect observations

Every successful `wfperf` command appends one row to `runs.csv`. The row has 14
backend-independent workflow inputs and eight measured outputs. With one
intermediate stage, the inputs map directly to the original source,
intermediate, and sink model. If a workflow has multiple intermediate stages,
their values are averaged by logical task count; `intermediate_count` is the
total number of intermediate tasks.

The database is local runtime state and is not committed to the repository.
[`database-template.csv`](database-template.csv) documents the complete schema.
Use `--no-ml-record` for a run that should not become training data.

## Train offline

After collecting observations, train the model explicitly:

```bash
wfperf-train
```

The command evaluates polynomial linear regression, a multilayer perceptron,
random forest, gradient boosting, and support vector regression with shuffled
cross-validation. Targets are standardized inside each training fold, and the
candidate with the lowest normalized cross-validation mean squared error is
fitted to all observations and saved.

Choose one algorithm or custom paths when needed:

```bash
wfperf-train /path/to/runs.csv \
    --output /path/to/model.joblib \
    --algorithm mlp
```

The trainer also accepts the historical 22-column CSV layout used by the early
prototype. Rows must be complete and numeric. A model artifact is a Python
serialization file; only load artifacts produced in a trusted environment.

## Validate every benchmark run

Once the model exists at the configured path, normal Wilkins and Parsl runs
predict all eight timing values and compare them with the measurements. WfPerf
prints one warning per metric whose relative difference is greater than 5%:

```text
WfPerf ML warning: sink_task_time_ms actual=..., predicted=..., difference=... (threshold=5.00%)
```

Validation details are also stored under `ml.validation` in `summary.json`.
Warnings use standard error, including when `--json` sends the summary to
standard output. Change the threshold with `--ml-threshold PERCENT`, select a
model with `--ml-model PATH`, or use `--no-ml-validation` for a deliberately
unvalidated run.

The implementation used by the installed commands is in `src/wfperf/ml/`.
The `ml/train.py` wrapper provides the same offline trainer from a source tree.
