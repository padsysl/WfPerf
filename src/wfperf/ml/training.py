# Copyright (c) 2026, University of Florida. All rights reserved.
#
# This file is part of WfPerf, developed by members of Prof. Xiaoyi Lu's
# group at the University of Florida. See LICENSE in the top-level directory.

"""Explicit, offline training command for WfPerf performance models."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from wfperf.ml.data import (
    DATABASE_COLUMNS,
    FEATURE_NAMES,
    TARGET_NAMES,
    default_database_path,
    default_model_path,
)
from wfperf.ml.inference import MODEL_SCHEMA_VERSION


ALGORITHMS = (
    "polynomial-linear",
    "mlp",
    "random-forest",
    "gradient-boosting",
    "svr",
)


def load_database(path: Path) -> Tuple[Any, Any]:
    """Load a canonical or legacy 22-column WfPerf database."""

    try:
        import numpy as np
    except ImportError as error:
        raise ImportError("model training requires the 'ml' installation extra") from error

    with Path(path).open(newline="") as stream:
        rows = list(csv.reader(stream))
    if not rows:
        raise ValueError("{} is empty".format(path))

    header = tuple(value.strip() for value in rows[0])
    canonical = header == DATABASE_COLUMNS
    legacy = len(header) == len(DATABASE_COLUMNS) and header[:5] == (
        "count",
        "nprocs",
        "sleep",
        "particle",
        "iter",
    )
    if not canonical and not legacy:
        raise ValueError(
            "{} does not use the canonical or legacy 22-column schema".format(path)
        )

    numeric_rows = []
    for line_number, row in enumerate(rows[1:], start=2):
        if not row or not any(value.strip() for value in row):
            continue
        if len(row) != len(DATABASE_COLUMNS):
            raise ValueError("{}:{} has {} columns; expected {}".format(
                path, line_number, len(row), len(DATABASE_COLUMNS)
            ))
        try:
            numeric_rows.append([float(value) for value in row])
        except ValueError as error:
            raise ValueError("{}:{} contains a non-numeric value".format(
                path, line_number
            )) from error
    if len(numeric_rows) < 2:
        raise ValueError("model training requires at least two observations")

    values = np.asarray(numeric_rows, dtype=float)
    return values[:, : len(FEATURE_NAMES)], values[:, len(FEATURE_NAMES) :]


def _make_estimator(name: str, random_state: int) -> Any:
    try:
        from sklearn.compose import TransformedTargetRegressor
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import PolynomialFeatures, StandardScaler
        from sklearn.svm import SVR
    except ImportError as error:
        raise ImportError("model training requires the 'ml' installation extra") from error

    if name == "polynomial-linear":
        regressor = Pipeline(
            [
                ("scale", StandardScaler()),
                ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
                ("regression", LinearRegression()),
            ]
        )
    elif name == "mlp":
        regressor = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "regression",
                    MLPRegressor(
                        hidden_layer_sizes=(100, 50),
                        max_iter=5000,
                        learning_rate="adaptive",
                        learning_rate_init=0.01,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    elif name == "random-forest":
        regressor = RandomForestRegressor(
            n_estimators=100, random_state=random_state, n_jobs=-1
        )
    elif name == "gradient-boosting":
        regressor = MultiOutputRegressor(
            GradientBoostingRegressor(n_estimators=100, random_state=random_state)
        )
    elif name == "svr":
        regressor = Pipeline(
            [
                ("scale", StandardScaler()),
                ("regression", MultiOutputRegressor(SVR(kernel="rbf"))),
            ]
        )
    else:
        raise ValueError("unsupported algorithm {!r}".format(name))

    return TransformedTargetRegressor(regressor=regressor, transformer=StandardScaler())


def _cross_validated_error(
    name: str, features: Any, targets: Any, folds: int, random_state: int
) -> float:
    import numpy as np
    from sklearn.base import clone
    from sklearn.model_selection import KFold

    splitter = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    fold_errors = []
    template = _make_estimator(name, random_state)
    for train_indices, test_indices in splitter.split(features):
        estimator = clone(template)
        estimator.fit(features[train_indices], targets[train_indices])
        predicted = estimator.predict(features[test_indices])
        scale = np.std(targets[train_indices], axis=0)
        scale[scale == 0.0] = 1.0
        fold_errors.append(float(np.mean(((predicted - targets[test_indices]) / scale) ** 2)))
    return float(np.mean(fold_errors))


def train(
    database: Path,
    output: Path,
    algorithm: str = "best",
    folds: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train, select, and atomically save one WfPerf model artifact."""

    if algorithm != "best" and algorithm not in ALGORITHMS:
        raise ValueError("unsupported algorithm {!r}".format(algorithm))
    features, targets = load_database(database)
    if folds < 2:
        raise ValueError("cross-validation folds must be at least two")
    folds = min(folds, len(features))
    candidates = ALGORITHMS if algorithm == "best" else (algorithm,)
    scores = {
        name: _cross_validated_error(name, features, targets, folds, random_state)
        for name in candidates
    }
    selected = min(scores, key=scores.get)
    estimator = _make_estimator(selected, random_state)
    estimator.fit(features, targets)

    artifact = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "algorithm": selected,
        "feature_names": FEATURE_NAMES,
        "target_names": TARGET_NAMES,
        "training_rows": int(len(features)),
        "cross_validation_folds": folds,
        "normalized_mse": scores,
        "estimator": estimator,
    }

    import joblib

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    joblib.dump(artifact, temporary)
    os.replace(str(temporary), str(output))
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a WfPerf performance model offline."
    )
    parser.add_argument(
        "database",
        nargs="?",
        type=Path,
        default=default_database_path(),
        help="observation CSV (default: WfPerf per-user runs.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_model_path(),
        help="model artifact path (default: WfPerf per-user model.joblib)",
    )
    parser.add_argument(
        "--algorithm",
        choices=("best",) + ALGORITHMS,
        default="best",
        help="model to train, or evaluate all candidates and keep the best",
    )
    parser.add_argument("--folds", type=int, default=5, help="cross-validation folds")
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        artifact = train(
            args.database,
            args.output,
            algorithm=args.algorithm,
            folds=args.folds,
            random_state=args.random_state,
        )
    except (ImportError, OSError, ValueError) as error:
        print("wfperf-train: {}".format(error), file=sys.stderr)
        return 2

    for name, score in sorted(artifact["normalized_mse"].items()):
        print("{} normalized CV MSE: {:.6g}".format(name, score))
    print("Selected: {}".format(artifact["algorithm"]))
    print("Model: {}".format(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
