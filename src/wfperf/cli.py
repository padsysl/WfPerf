# Copyright (c) 2026, University of Florida. All rights reserved.
#
# This file is part of WfPerf, developed by members of Prof. Xiaoyi Lu's
# group at the University of Florida. See LICENSE in the top-level directory.

"""Backend-neutral command-line interface for WfPerf."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from wfperf.config import ConfigError, load_workflow
from wfperf.ml.data import (
    append_observation,
    default_database_path,
    default_model_path,
    extract_features,
    extract_targets,
)


def _nonnegative_percent(value: str) -> float:
    result = float(value)
    if result < 0:
        raise argparse.ArgumentTypeError("percentage must be non-negative")
    return result


def _load_parsl_config(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("wfperf_user_parsl_config", str(path))
    if spec is None or spec.loader is None:
        raise ValueError("cannot import Parsl configuration from {}".format(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "get_config"):
        return module.get_config()
    if hasattr(module, "CONFIG"):
        return module.CONFIG
    raise ValueError("{} must define get_config() or CONFIG".format(path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a WfPerf workflow with a selected execution backend."
    )
    parser.add_argument("config", type=Path, help="WfPerf YAML configuration")
    parser.add_argument(
        "--backend",
        choices=("parsl", "wilkins"),
        required=True,
        help="workflow backend to use",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("wfperf-runs"),
        help="directory in which to create the run directory",
    )
    parser.add_argument(
        "--keep-files",
        action="store_true",
        help="keep generated HDF5 files after metrics are collected",
    )
    parser.add_argument(
        "--json", action="store_true", help="print the complete summary as JSON"
    )

    modeling = parser.add_argument_group("Performance model")
    modeling.add_argument(
        "--ml-database",
        type=Path,
        default=default_database_path(),
        help="CSV to which completed-run observations are appended",
    )
    modeling.add_argument(
        "--ml-model",
        type=Path,
        default=default_model_path(),
        help="offline-trained model used to validate completed runs",
    )
    modeling.add_argument(
        "--ml-threshold",
        type=_nonnegative_percent,
        default=5.0,
        metavar="PERCENT",
        help="warn when measured and predicted timing differ by more than this percent",
    )
    modeling.add_argument(
        "--no-ml-record",
        action="store_true",
        help="do not append this run to the ML observation database",
    )
    modeling.add_argument(
        "--no-ml-validation",
        action="store_true",
        help="do not validate this run with the offline-trained model",
    )

    parsl = parser.add_argument_group("Parsl backend")
    parsl.add_argument(
        "--parsl-config",
        type=Path,
        help="Python file defining get_config() or CONFIG",
    )
    parsl.add_argument(
        "--max-workers",
        type=int,
        help="maximum thread-pool workers when --parsl-config is not supplied",
    )

    wilkins = parser.add_argument_group("Wilkins backend")
    wilkins.add_argument(
        "--wilkins-runtime",
        type=Path,
        default=(
            Path(os.environ["WFPERF_WILKINS_RUNTIME"])
            if "WFPERF_WILKINS_RUNTIME" in os.environ
            else None
        ),
        help=(
            "directory containing the built WfPerf .hx modules "
            "(default: WFPERF_WILKINS_RUNTIME)"
        ),
    )
    wilkins.add_argument(
        "--wilkins-driver",
        default="wilkins-master.py",
        help="installed Wilkins driver command or path (default: wilkins-master.py)",
    )
    wilkins.add_argument(
        "--wilkins-launcher",
        default="mpirun",
        help="MPI launcher executable (default: mpirun)",
    )
    wilkins.add_argument(
        "--wilkins-launcher-args",
        default="-l",
        help="shell-style launcher arguments before -n (default: -l)",
    )
    return parser


def _create_backend(args: argparse.Namespace):
    if args.backend == "parsl":
        if args.wilkins_runtime is not None:
            raise ValueError("--wilkins-runtime is only valid with --backend wilkins")
        from wfperf.backends.parsl.backend import ParslBackend

        config = (
            _load_parsl_config(args.parsl_config)
            if args.parsl_config is not None
            else None
        )
        return ParslBackend(config=config, max_workers=args.max_workers)

    if args.parsl_config is not None or args.max_workers is not None:
        raise ValueError("Parsl options are only valid with --backend parsl")
    if args.wilkins_runtime is None:
        raise ValueError("--wilkins-runtime is required with --backend wilkins")

    from wfperf.backends.wilkins.backend import WilkinsBackend

    return WilkinsBackend(
        runtime_directory=args.wilkins_runtime,
        launcher=args.wilkins_launcher,
        launcher_args=shlex.split(args.wilkins_launcher_args),
        driver=args.wilkins_driver,
    )


def _write_summary(result: Any) -> None:
    temporary = result.summary_path.with_name(result.summary_path.name + ".tmp")
    temporary.write_text(json.dumps(result.summary, indent=2, sort_keys=True) + "\n")
    temporary.replace(result.summary_path)


def _process_ml(workflow: Any, result: Any, args: argparse.Namespace) -> None:
    """Record one run and, when available, compare it with the trained model."""

    features = extract_features(workflow)
    targets = extract_targets(result.summary)
    ml_summary = {
        "database": str(args.ml_database),
        "recorded": False,
        "validation": {"status": "disabled"},
    }

    if not args.no_ml_record:
        try:
            append_observation(args.ml_database, features, targets)
            ml_summary["recorded"] = True
        except (OSError, ValueError) as error:
            ml_summary["record_error"] = str(error)
            print(
                "WfPerf ML warning: could not record this run: {}".format(error),
                file=sys.stderr,
            )

    if not args.no_ml_validation:
        if not args.ml_model.is_file():
            ml_summary["validation"] = {
                "status": "model_not_found",
                "model": str(args.ml_model),
            }
        else:
            try:
                from wfperf.ml.inference import validate_run

                validation = validate_run(
                    args.ml_model,
                    features,
                    targets,
                    threshold_percent=args.ml_threshold,
                )
                ml_summary["validation"] = validation
                for violation in validation["violations"]:
                    difference = violation["difference_percent"]
                    difference_text = (
                        "unbounded"
                        if difference is None
                        else "{:.2f}%".format(difference)
                    )
                    print(
                        "WfPerf ML warning: {} actual={:.6g}, predicted={:.6g}, "
                        "difference={} (threshold={:.2f}%)".format(
                            violation["metric"],
                            violation["actual"],
                            violation["predicted"],
                            difference_text,
                            args.ml_threshold,
                        ),
                        file=sys.stderr,
                    )
            except (ImportError, OSError, ValueError) as error:
                ml_summary["validation"] = {
                    "status": "error",
                    "model": str(args.ml_model),
                    "message": str(error),
                }
                print(
                    "WfPerf ML warning: model validation failed: {}".format(error),
                    file=sys.stderr,
                )

    result.summary["ml"] = ml_summary
    _write_summary(result)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        workflow = load_workflow(args.config)
        backend = _create_backend(args)
        result = backend.run(
            workflow,
            output_directory=args.output_dir,
            keep_files=args.keep_files,
        )
        _process_ml(workflow, result, args)
    except (ConfigError, ImportError, OSError, RuntimeError, ValueError) as error:
        print("wfperf: {}".format(error), file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result.summary, indent=2, sort_keys=True))
    else:
        print("Backend: {}".format(result.summary["backend"]))
        print("Tasks: {}".format(len(result.summary["tasks"])))
        print("End-to-end time: {:.3f} s".format(result.summary["e2e_time_seconds"]))
        print("Summary: {}".format(result.summary_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
