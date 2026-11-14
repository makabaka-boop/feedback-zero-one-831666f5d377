"""SQLite persistence for the smoother service.

A single database file holds every durable fact:

* the validated model and the current input vector ``y``;
* the monotonic input ``version`` and the revision log (one row per accepted
  revision, keyed by the client-supplied alphanumeric ``id``);
* the worker generation token (an ever-increasing integer claimed by each
  starting worker, which invalidates every older worker);
* the last fully published snapshot (or no row until the first publish).

All state transitions run inside ``BEGIN IMMEDIATE`` transactions, so an
accepted revision, a generation claim and a snapshot publish are each atomic
with respect to crashes and to other connections.
"""

from __future__ import annotations

import json
import sqlite3
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple

from ..validate import ValidationError, load_input
from . import crash
from .revisions import canonical_changes, check_horizon

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    id      INTEGER PRIMARY KEY CHECK (id = 1),
    model   TEXT NOT NULL,
    y       TEXT NOT NULL,
    version INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS revisions (
    seq     INTEGER PRIMARY KEY AUTOINCREMENT,
    rev_id  TEXT NOT NULL UNIQUE,
    version INTEGER NOT NULL,
    changes TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS generation (
    id         INTEGER PRIMARY KEY CHECK (id = 1),
    generation INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS published (
    id       INTEGER PRIMARY KEY CHECK (id = 1),
    version  INTEGER NOT NULL,
    snapshot TEXT NOT NULL
);
"""


class ConflictError(Exception):
    """HTTP 409: the same revision id was already used with other content."""

    def __init__(self, existing_version: int):
        super().__init__(f"revision id already exists (version "
                         f"{existing_version})")
        self.existing_version = existing_version

    def payload(self) -> dict:
        return {"error": {
            "path": "$.id",
            "message": ("revision id already exists with different content; "
                        f"stored version is {self.existing_version}"),
        }}


# ---------------------------------------------------------------------------
# JSON <-> exact rational helpers (mirrors smoother.core._num)
# ---------------------------------------------------------------------------
def _num(value: Fraction) -> str:
    value = Fraction(value)
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _y_to_jsonable(y: List[Optional[Fraction]]) -> List[Optional[str]]:
    return [None if item is None else _num(item) for item in y]


def _y_from_jsonable(raw: Any, path: str = "$.y") -> List[Optional[Fraction]]:
    if not isinstance(raw, list):
        raise ValidationError(path, "stored y must be an array")
    out: List[Optional[Fraction]] = []
    for index, item in enumerate(raw):
        if item is None:
            out.append(None)
        else:
            out.append(Fraction(item))
    return out


def model_to_jsonable(inp: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize a validated model's eight members to plain JSON values."""
    return {
        "a": [_num(v) for v in inp["a"]],
        "P": [[_num(v) for v in row] for row in inp["P"]],
        "D": [[_num(v) for v in row] for row in inp["D"]],
        "F": [[_num(v) for v in row] for row in inp["F"]],
        "Q": [[_num(v) for v in row] for row in inp["Q"]],
        "h": [_num(v) for v in inp["h"]],
        "r": _num(inp["r"]),
        "y": _y_to_jsonable(inp["y"]),
    }


def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30.0, isolation_level=None)
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def initialize(conn: sqlite3.Connection, model_text: str) -> None:
    """Create the schema and bootstrap version 0 from a legal model.

    Fails (leaving any existing database untouched) when the model is
    invalid or the database is already initialized.
    """
    inp = load_input(model_text)
    stored = model_to_jsonable(inp)
    model_blob = json.dumps(
        {k: stored[k] for k in ("a", "P", "D", "F", "Q", "h", "r")},
        separators=(",", ":"), ensure_ascii=False, sort_keys=True)
    y_blob = json.dumps(stored["y"], separators=(",", ":"),
                        ensure_ascii=False)

    conn.executescript(SCHEMA)
    conn.execute("BEGIN IMMEDIATE")
    try:
        existing = conn.execute("SELECT 1 FROM meta WHERE id = 1").fetchone()
        if existing is not None:
            raise ValidationError("$", "database is already initialized")
        conn.execute(
            "INSERT INTO meta (id, model, y, version) VALUES (1, ?, ?, 0)",
            (model_blob, y_blob))
        conn.execute(
            "INSERT INTO generation (id, generation) VALUES (1, 0)")
        conn.commit()
    except Exception:
        conn.rollback()
        raise


# ---------------------------------------------------------------------------
# revision acceptance
# ---------------------------------------------------------------------------
def _stored_changes(conn: sqlite3.Connection, rev_id: str
                    ) -> Optional[Tuple[int, List[Tuple[int, Any]]]]:
    row = conn.execute(
        "SELECT version, changes FROM revisions WHERE rev_id = ?",
        (rev_id,)).fetchone()
    if row is None:
        return None
    version, blob = row
    raw = json.loads(blob)
    pairs: List[Tuple[int, Any]] = []
    for time_value, value in raw:
        pairs.append((time_value, None if value is None else Fraction(value)))
    return version, pairs


def submit_revision(conn: sqlite3.Connection,
                    rev_id: str,
                    changes: List[Tuple[int, Any]]
                    ) -> int:
    """Validate-then-accept a revision atomically.

    Returns the integer version associated with the revision.  A previously
    unseen id updates ``y`` atomically and increments the version; an id seen
    before with identical content returns its stored version (no state
    change); an id seen before with different content raises
    :class:`ConflictError`.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        y_blob = conn.execute("SELECT y FROM meta WHERE id = 1").fetchone()[0]
        y = _y_from_jsonable(json.loads(y_blob))
        check_horizon(changes, len(y))

        prior = _stored_changes(conn, rev_id)
        if prior is not None:
            existing_version, existing_changes = prior
            if canonical_changes(existing_changes) == \
                    canonical_changes(changes):
                conn.rollback()
                return existing_version
            conn.rollback()
            raise ConflictError(existing_version)

        for time_value, value in changes:
            y[time_value] = value

        new_version = conn.execute(
            "SELECT version FROM meta WHERE id = 1").fetchone()[0] + 1
        stored_pairs = [
            (time_value, None if value is None else _num(value))
            for time_value, value in changes]
        conn.execute(
            "UPDATE meta SET y = ?, version = ? WHERE id = 1",
            (json.dumps(_y_to_jsonable(y), separators=(",", ":"),
                        ensure_ascii=False), new_version))
        conn.execute(
            "INSERT INTO revisions (rev_id, version, changes) "
            "VALUES (?, ?, ?)",
            (rev_id, new_version,
             json.dumps(stored_pairs, separators=(",", ":"),
                        ensure_ascii=False)))
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    # The row is durable now; kill here to prove receipts are reconstructible
    # from the log rather than from any in-memory server state.
    crash.crash_point(crash.AFTER_ACCEPT)
    return new_version


# ---------------------------------------------------------------------------
# worker generation
# ---------------------------------------------------------------------------
def claim_generation(conn: sqlite3.Connection) -> int:
    """Atomically take a fresh, strictly increasing generation token."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute(
            "SELECT generation FROM generation WHERE id = 1").fetchone()
        new_generation = row[0] + 1
        conn.execute(
            "UPDATE generation SET generation = ? WHERE id = 1",
            (new_generation,))
        conn.commit()
        return new_generation
    except Exception:
        conn.rollback()
        raise


def current_generation(conn: sqlite3.Connection) -> int:
    return conn.execute(
        "SELECT generation FROM generation WHERE id = 1").fetchone()[0]


# ---------------------------------------------------------------------------
# publishing
# ---------------------------------------------------------------------------
def publish_snapshot(conn: sqlite3.Connection,
                     worker_generation: int,
                     computed_version: int,
                     snapshot: Dict[str, Any]) -> bool:
    """Publish a computed snapshot if it still represents the latest input.

    The publish commits only when the worker still owns the current
    generation *and* the input version it computed equals the latest input
    version.  Returns whether the snapshot was published.  Stale generations
    and stale computations are ignored silently.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        live_generation, latest_version = conn.execute(
            "SELECT g.generation, m.version "
            "FROM generation AS g, meta AS m WHERE g.id = 1 AND m.id = 1"
        ).fetchone()
        if live_generation != worker_generation \
                or latest_version != computed_version:
            conn.rollback()
            return False

        crash.crash_point(crash.BEFORE_PUBLISH_COMMIT)
        blob = json.dumps(snapshot, separators=(",", ":"),
                          ensure_ascii=False, sort_keys=True)
        conn.execute(
            "INSERT INTO published (id, version, snapshot) "
            "VALUES (1, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET version = excluded.version, "
            "snapshot = excluded.snapshot",
            (computed_version, blob))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    crash.crash_point(crash.AFTER_PUBLISH_COMMIT)
    return True


# ---------------------------------------------------------------------------
# reads
# ---------------------------------------------------------------------------
def get_query(conn: sqlite3.Connection) -> Dict[str, Any]:
    """Atomically return ``{"version": v, "published": snapshot-or-null}``.

    A non-null snapshot always comes from the same version it carries and
    contains exactly ``version``, ``y``, ``mean``, ``A`` and ``B``.
    """
    conn.execute("BEGIN")
    try:
        version = conn.execute(
            "SELECT version FROM meta WHERE id = 1").fetchone()[0]
        row = conn.execute(
            "SELECT version, snapshot FROM published WHERE id = 1"
        ).fetchone()
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    if row is None:
        return {"version": version, "published": None}
    published_version, blob = row
    snapshot = json.loads(blob)
    # Defensive invariant: a stored snapshot is always self-describing.
    assert snapshot["version"] == published_version
    return {"version": version, "published": snapshot}


def load_worker_input(conn: sqlite3.Connection
                      ) -> Tuple[int, int, Dict[str, Any]]:
    """Return ``(generation, version, model)`` for a worker computation.

    The model is rebuilt into the exact dict shape consumed by
    :func:`smoother.core.solve`, re-validated through the strict loader.
    """
    generation = current_generation(conn)
    model_blob, y_blob, version = conn.execute(
        "SELECT model, y, version FROM meta WHERE id = 1").fetchone()
    model = json.loads(model_blob)
    model["y"] = json.loads(y_blob)
    inp = load_input(json.dumps(model, ensure_ascii=False))
    return generation, version, inp
