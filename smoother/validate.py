"""Strict UTF-8 JSON input validation for the limiting smoother.

Every scalar must be a rational written as an ``integer`` or ``p/q`` string
with a strictly positive denominator.  Errors report a JSON path with
zero-based indices; pure syntax errors use the path ``"$"``.
"""

from __future__ import annotations

import json
from fractions import Fraction
from typing import Any, List, Tuple

from . import linalg

REQUIRED_KEYS = ("a", "P", "D", "F", "Q", "h", "r", "y")


class ValidationError(Exception):
    """Raised with a ready-to-serialize JSON error payload."""

    def __init__(self, path: str, message: str):
        super().__init__(f"{path}: {message}")
        self.path = path
        self.message = message

    def payload(self) -> dict:
        return {"error": {"path": self.path, "message": self.message}}


class _DuplicateChecker(dict):
    """dict subclass used as object_pairs_hook to reject repeated members."""

    def __init__(self, pairs):
        super().__init__()
        seen = set()
        for key, value in pairs:
            if key in seen:
                raise ValidationError("$", f"duplicate object member '{key}'")
            seen.add(key)
            self[key] = value


# ---------------------------------------------------------------------------
# scalar parsing
# ---------------------------------------------------------------------------
def parse_fraction(value: Any, path: str) -> Fraction:
    if not isinstance(value, str):
        raise ValidationError(
            path, "expected rational string 'p/q' with positive denominator")
    text = value
    if not text or any(ch.isspace() for ch in text):
        raise ValidationError(path, "malformed rational string")
    parts = text.split("/")
    if len(parts) > 2:
        raise ValidationError(path, "malformed rational string")
    try:
        if len(parts) == 1:
            numerator = int(parts[0], 10)
            denominator = 1
        else:
            numerator = int(parts[0], 10)
            denominator = int(parts[1], 10)
    except ValueError:
        raise ValidationError(path, "malformed rational string")
    if denominator <= 0:
        raise ValidationError(path, "denominator must be a positive integer")
    return Fraction(numerator, denominator)


def _parse_vector(value: Any, path: str) -> List[Fraction]:
    if not isinstance(value, list):
        raise ValidationError(path, "expected an array")
    return [parse_fraction(item, f"{path}[{i}]")
            for i, item in enumerate(value)]


def _parse_matrix(value: Any, path: str
                  ) -> Tuple[List[List[Fraction]], int, int]:
    if not isinstance(value, list):
        raise ValidationError(path, "expected a matrix array")
    rows: List[List[Fraction]] = []
    width = None
    for i, row in enumerate(value):
        if not isinstance(row, list):
            raise ValidationError(f"{path}[{i}]", "expected an array row")
        parsed = [parse_fraction(x, f"{path}[{i}][{j}]")
                  for j, x in enumerate(row)]
        if width is None:
            width = len(parsed)
        elif len(parsed) != width:
            raise ValidationError(f"{path}[{i}]",
                                  "ragged matrix: row length mismatch")
        rows.append(parsed)
    return rows, len(rows), width if width is not None else 0


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def load_input(text: str) -> dict:
    """Parse and fully validate the raw UTF-8 JSON text."""
    try:
        raw = json.loads(text, object_pairs_hook=_DuplicateChecker)
    except ValidationError:
        raise
    except json.JSONDecodeError:
        raise ValidationError("$", "malformed JSON syntax")

    if not isinstance(raw, dict):
        raise ValidationError("$", "top-level value must be an object")
    for key in raw:
        if key not in REQUIRED_KEYS:
            raise ValidationError(f"$.{key}", "unknown object member")
    missing = [k for k in REQUIRED_KEYS if k not in raw]
    if missing:
        raise ValidationError("$",
                              f"missing required member '{missing[0]}'")

    a = _parse_vector(raw["a"], "$.a")
    n = len(a)
    if n < 1:
        raise ValidationError("$.a", "must contain at least one element")
    h = _parse_vector(raw["h"], "$.h")
    if len(h) != n:
        raise ValidationError("$.h", f"length must be {n}")

    def square(name: str):
        mat, rows, cols = _parse_matrix(raw[name], f"$.{name}")
        if rows != n or cols != n:
            raise ValidationError(f"$.{name}",
                                  f"must be an {n}x{n} matrix")
        if not linalg.is_symmetric(mat):
            raise ValidationError(f"$.{name}", "matrix must be symmetric")
        if not linalg.is_symmetric_psd(mat):
            raise ValidationError(f"$.{name}",
                                  "matrix must be positive semidefinite")
        return mat

    p_mat = square("P")
    q_mat = square("Q")
    f_mat, f_rows, f_cols = _parse_matrix(raw["F"], "$.F")
    if f_rows != n or f_cols != n:
        raise ValidationError("$.F", f"must be an {n}x{n} matrix")

    # D: n rows, d columns, 0 <= d <= n, full column rank; d == 0 uses
    # n empty arrays.
    if not isinstance(raw["D"], list):
        raise ValidationError("$.D", "expected a matrix array")
    if len(raw["D"]) != n:
        raise ValidationError("$.D", f"must have exactly {n} rows")
    d_rows: List[List[Fraction]] = []
    d_width = None
    for i, row in enumerate(raw["D"]):
        if not isinstance(row, list):
            raise ValidationError(f"$.D[{i}]", "expected an array row")
        parsed = [parse_fraction(x, f"$.D[{i}][{j}]")
                  for j, x in enumerate(row)]
        if d_width is None:
            d_width = len(parsed)
        elif len(parsed) != d_width:
            raise ValidationError(f"$.D[{i}]",
                                  "ragged matrix: row length mismatch")
        d_rows.append(parsed)
    d = d_width if d_width is not None else 0
    if d > n:
        raise ValidationError("$.D",
                              f"column count {d} exceeds state size {n}")
    if d == 0:
        if any(row for row in d_rows):
            raise ValidationError("$.D",
                                  "d=0 requires n empty row arrays")
    else:
        if linalg.rank(d_rows) != d:
            raise ValidationError("$.D",
                                  "columns must be linearly independent "
                                  "(full column rank)")

    r = parse_fraction(raw["r"], "$.r")
    if r <= 0:
        raise ValidationError("$.r", "must be strictly positive")

    y_raw = raw["y"]
    if not isinstance(y_raw, list) or len(y_raw) == 0:
        raise ValidationError("$.y", "must be a non-empty array")
    y: List[Fraction | None] = []
    for i, item in enumerate(y_raw):
        if item is None:
            y.append(None)
        elif isinstance(item, (bool, int, float)):
            raise ValidationError(
                f"$.y[{i}]",
                "expected rational string or null, not a bare number")
        else:
            y.append(parse_fraction(item, f"$.y[{i}]"))

    return {
        "a": a, "P": p_mat, "D": d_rows, "F": f_mat, "Q": q_mat,
        "h": h, "r": r, "y": y, "n": n, "d": d, "T": len(y),
    }
