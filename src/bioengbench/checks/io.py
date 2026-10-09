"""Strict readers for submission outputs. A schema error is a failed check."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class SchemaError(ValueError):
    pass


def read_csv_columns(path: Path, columns: tuple[str, ...]) -> dict[str, FloatArray]:
    """Read a CSV whose header must equal `columns` exactly, in order.

    Every cell must parse as a finite float. Rejecting rather than repairing is
    deliberate: a grader that tolerates sloppy files teaches sloppy files.
    """
    with path.open(newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        raise SchemaError(f"{path.name} is empty")
    header = tuple(cell.strip() for cell in rows[0])
    if header != columns:
        raise SchemaError(f"{path.name}: expected header {columns}, got {header}")
    body = rows[1:]
    if not body:
        raise SchemaError(f"{path.name} has a header but no data rows")
    data = np.empty((len(body), len(columns)))
    for i, row in enumerate(body):
        if len(row) != len(columns):
            raise SchemaError(f"{path.name}: row {i + 2} has {len(row)} fields")
        data[i] = [float(cell) for cell in row]
    if not np.all(np.isfinite(data)):
        raise SchemaError(f"{path.name} contains NaN or infinite values")
    return {name: data[:, j] for j, name in enumerate(columns)}


def read_json_numbers(path: Path, keys: tuple[str, ...]) -> dict[str, float]:
    """Read a JSON object that must contain every key with a finite number."""
    obj = json.loads(path.read_text())
    if not isinstance(obj, dict):
        raise SchemaError(f"{path.name}: top level must be an object")
    out: dict[str, float] = {}
    for key in keys:
        value = obj.get(key)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise SchemaError(f"{path.name}: '{key}' must be a number, got {value!r}")
        if not math.isfinite(value):
            raise SchemaError(f"{path.name}: '{key}' is not finite")
        out[key] = float(value)
    return out
