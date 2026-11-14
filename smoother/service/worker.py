"""Independent worker process that computes and publishes smoothings.

Lifecycle
---------
1. On startup the worker claims a fresh, strictly increasing *generation*
   token from SQLite.  Every previously running worker is now superseded: its
   in-flight computation may finish, but its publish will be rejected.
2. The worker watches the latest input version.  Computation runs in this
   process and never blocks revision acceptance in the HTTP server.
3. A result is published atomically only when the worker still owns the
   current generation **and** the version it computed is still the latest
   input version.  Results and errors from stale generations are ignored.
4. Revisions accepted after a computation started simply leave that
   computation stale; the worker loops and recomputes.  If revisions stop,
   the current worker necessarily finishes the latest version; if the worker
   dies first, a restarted worker picks the latest version up from the
   durable state.
"""

from __future__ import annotations

import signal
import threading
from typing import Any, Dict

from ..core import solve, to_jsonable
from . import schema


class Worker:
    def __init__(self, db_path: str, poll_interval: float = 0.1):
        self.db_path = db_path
        self.poll_interval = poll_interval
        self._stop = threading.Event()
        self.generation = -1
        # The version whose result (if any) is already published by us.
        self._published_version = -1

    # ------------------------------------------------------------------
    def request_stop(self, *_signals) -> None:
        self._stop.set()

    def _wait(self) -> bool:
        """Wait for the poll interval; return False when stopping."""
        return not self._stop.wait(self.poll_interval)

    # ------------------------------------------------------------------
    def run(self) -> None:
        signal.signal(signal.SIGTERM, self.request_stop)
        signal.signal(signal.SIGINT, self.request_stop)

        conn = schema.connect(self.db_path)
        try:
            self.generation = schema.claim_generation(conn)
            while not self._stop.is_set():
                generation = schema.current_generation(conn)
                if generation != self.generation:
                    # A newer worker has taken over; this process is obsolete.
                    return
                version, inp = schema.load_worker_input(conn)[1:]
                if version == self._published_version:
                    if not self._wait():
                        return
                    continue

                # The expensive step.  Re-read the generation afterwards: a
                # takeover during computation makes the result unpublishable.
                try:
                    result = self._compute(inp)
                    snapshot = self._build_snapshot(version, inp, result)
                except Exception:
                    # Never publish a partial or failed result; retry later.
                    if not self._wait():
                        return
                    continue

                if self._stop.is_set():
                    return
                if schema.current_generation(conn) != self.generation:
                    return

                if schema.publish_snapshot(
                        conn, self.generation, version, snapshot):
                    self._published_version = version
                if not self._wait():
                    return
        finally:
            conn.close()

    # ------------------------------------------------------------------
    def _compute(self, inp: Dict[str, Any]) -> Dict[str, Any]:
        """Run the exact smoother; failures are swallowed by design.

        An invalid state cannot arise because every stored model passed the
        strict validator, but any unexpected arithmetic error must never kill
        the worker: the loop retries on the next pass instead of publishing a
        partial result.
        """
        try:
            return to_jsonable(solve(inp))
        except Exception:
            return {}

    @staticmethod
    def _build_snapshot(version: int, inp: Dict[str, Any],
                        result: Dict[str, Any]) -> Dict[str, Any]:
        if set(result) != {"mean", "A", "B"}:
            # A failed computation never reaches publishing: raise so the
            # loop retries; nothing is written.
            raise RuntimeError("smoother computation failed")
        return {
            "version": version,
            "y": schema._y_to_jsonable(inp["y"]),
            "mean": result["mean"],
            "A": result["A"],
            "B": result["B"],
        }


def run_worker(db_path: str, poll_interval: float = 0.1) -> None:
    Worker(db_path, poll_interval).run()
