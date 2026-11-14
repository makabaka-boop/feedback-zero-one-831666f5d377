"""Strict validation of revision requests.

A revision document contains exactly two members::

    {"id": "<ASCII alphanumeric, non-empty>",
     "changes": [[time, value], ...]}

* ``time`` is a non-negative JSON integer (booleans are rejected as booleans,
  not accepted as integers);
* ``value`` follows the same rational-string-or-null rule as the model's
  ``y``;
* the ``changes`` array may be empty; duplicate times and times outside the
  model horizon are rejected.

Error payloads reuse the project convention
``{"error": {"path": ..., "message": ...}}`` with zero-based indices; pure
syntax errors use the path ``"$"``.
"""

from __future__ import annotations

import json
from typing import Any, List, Tuple

from ..validate import ValidationError, _DuplicateChecker, parse_fraction


def parse_revision(text: str) -> Tuple[str, List[Tuple[int, Any]]]:
    """Parse raw JSON text into ``(id, canonical_changes)``.

    ``canonical_changes`` is a list of ``(time, value)`` pairs with values
    reduced to canonical form: a ``fractions.Fraction`` or ``None``.  The
    pairs keep the request's item order; equality of two revisions ignores
    that order and compares numerically.
    """
    try:
        raw = json.loads(text, object_pairs_hook=_DuplicateChecker)
    except ValidationError:
        raise
    except json.JSONDecodeError:
        raise ValidationError("$", "malformed JSON syntax")

    if not isinstance(raw, dict):
        raise ValidationError("$", "revision must be an object")

    for key in raw:
        if key not in ("id", "changes"):
            raise ValidationError(f"$.{key}", "unknown object member")
    if "id" not in raw:
        raise ValidationError("$.id", "missing required member 'id'")
    if "changes" not in raw:
        raise ValidationError("$.changes",
                              "missing required member 'changes'")

    ident = raw["id"]
    if not isinstance(ident, str) or not ident:
        raise ValidationError("$.id",
                              "must be a non-empty ASCII alphanumeric string")
    try:
        ident.encode("ascii")
    except UnicodeEncodeError:
        raise ValidationError("$.id",
                              "must be a non-empty ASCII alphanumeric string")
    if not ident.isalnum():
        # str.isalnum also admits Unicode; the ASCII encode above plus this
        # check restricts to ASCII letters and digits.
        raise ValidationError("$.id",
                              "must be a non-empty ASCII alphanumeric string")

    changes_raw = raw["changes"]
    if not isinstance(changes_raw, list):
        raise ValidationError("$.changes", "must be an array")

    seen_times = set()
    changes: List[Tuple[int, Any]] = []
    for index, item in enumerate(changes_raw):
        base = f"$.changes[{index}]"
        if not isinstance(item, list):
            raise ValidationError(base, "each change must be [time, value]")
        if len(item) != 2:
            raise ValidationError(base,
                                  "each change must be exactly [time, value]")
        time_value, y_value = item

        # bool is a subclass of int: reject it explicitly.
        if isinstance(time_value, bool) or not isinstance(time_value, int):
            raise ValidationError(f"{base}[0]",
                                  "time must be a non-negative integer")
        if time_value < 0:
            raise ValidationError(f"{base}[0]",
                                  "time must be a non-negative integer")
        if time_value in seen_times:
            raise ValidationError(f"{base}[0]",
                                  f"duplicate time {time_value} in changes")
        seen_times.add(time_value)

        if y_value is None:
            value = None
        elif isinstance(y_value, bool):
            raise ValidationError(
                f"{base}[1]",
                "expected rational string or null, not a boolean")
        else:
            value = parse_fraction(y_value, f"{base}[1]")
        changes.append((time_value, value))

    return ident, changes


def check_horizon(changes: List[Tuple[int, Any]], horizon: int) -> None:
    """Reject times that do not index an element of the model's ``y``."""
    for index, (time_value, _) in enumerate(changes):
        if time_value >= horizon:
            raise ValidationError(
                f"$.changes[{index}][0]",
                f"time {time_value} is out of bounds for horizon {horizon}")


def canonical_changes(changes: List[Tuple[int, Any]]
                      ) -> List[Tuple[int, Any]]:
    """Order-independent canonical form, used for same-id comparison."""
    return sorted(changes, key=lambda pair: pair[0])
