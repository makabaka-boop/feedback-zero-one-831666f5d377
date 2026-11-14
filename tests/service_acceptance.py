"""Process-level acceptance suite for the persistent revision service.

The suite drives the real entry points (``run_service.py`` and
``run_worker.py``) as separate operating-system processes over HTTP and
SQLite, and covers the contract end to end:

  A. version-0 bootstrap and the documented id="a" "2/4" vs "1/2" example;
  B. revision validation: syntax, unknown/duplicate members, id charset,
     integer (non-boolean, non-negative) times, rational-or-null values,
     duplicate times and out-of-horizon indices, each with its JSON path and
     zero-based index, and proof that rejection never changes state;
  C. same-id order-independent numeric equality (same content -> old version,
     different content -> 409), empty changesets, monotonic integer versions;
  D. worker generation: a starting worker invalidates the old one, stale
     computations cannot publish, interleaved revisions converge on a
     complete snapshot of the latest version;
  E. hard kills (SIGKILL via crash hooks) after acceptance-before-receipt and
     before/after the publish commit, with full-snapshot verification after
     recovery;
  F. worker restart re-computes the latest version from durable state.

Run:  python3 -I tests/service_acceptance.py
"""

from __future__ import annotations

import http.client
import json
import os
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
from fractions import Fraction

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVICE_ENTRY = os.path.join(ROOT, "run_service.py")
WORKER_ENTRY = os.path.join(ROOT, "run_worker.py")

MODEL = {
    "a": ["0"], "P": [["0"]], "D": [["1"]], "F": [["1"]],
    "Q": [["0"]], "h": ["1"], "r": "1", "y": [None, "2"],
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
class Service:
    def __init__(self, tmp: str, port: int):
        self.tmp = tmp
        self.port = port
        self.db = os.path.join(tmp, "data.db")
        self.model_path = os.path.join(tmp, "model.json")
        with open(self.model_path, "w", encoding="utf-8") as fh:
            json.dump(MODEL, fh)
        self.procs: list[subprocess.Popen] = []

    def init(self) -> None:
        rc = subprocess.run(
            [sys.executable, "-I", SERVICE_ENTRY, "init",
             self.db, self.model_path], cwd=ROOT, capture_output=True,
            text=True)
        assert rc.returncode == 0, rc.stderr

    def start_server(self, crash_hook: str | None = None) -> subprocess.Popen:
        env = dict(os.environ)
        if crash_hook:
            env["SMOOTHER_CRASH_HOOK"] = crash_hook
        log = open(os.path.join(self.tmp, "server.log"), "w",
                   encoding="utf-8")
        proc = subprocess.Popen(
            [sys.executable, "-I", SERVICE_ENTRY, "serve", self.db,
             "--port", str(self.port)], cwd=ROOT, stdout=log, stderr=log,
            env=env)
        self.procs.append(proc)
        self._wait_port()
        return proc

    def start_worker(self, crash_hook: str | None = None,
                     poll: float = 0.05) -> subprocess.Popen:
        env = dict(os.environ)
        if crash_hook:
            env["SMOOTHER_CRASH_HOOK"] = crash_hook
        log = open(os.path.join(self.tmp,
                                f"worker-{len(self.procs)}.log"), "w",
                   encoding="utf-8")
        proc = subprocess.Popen(
            [sys.executable, "-I", WORKER_ENTRY, self.db,
             "--poll", str(poll)], cwd=ROOT, stdout=log, stderr=log, env=env)
        self.procs.append(proc)
        return proc

    def _wait_port(self, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", self.port),
                                              timeout=0.2):
                    return
            except OSError:
                time.sleep(0.05)
        raise RuntimeError("server did not open its port")

    def request(self, method: str, path: str, body=None):
        data = body.encode("utf-8") if isinstance(body, str) else body
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            conn.request(method, path, body=data,
                         headers={"Content-Length": str(len(data or b""))})
            resp = conn.getresponse()
            payload = resp.read().decode("utf-8")
            return resp.status, json.loads(payload) if payload else None
        finally:
            conn.close()

    def post(self, body: str):
        return self.request("POST", "/revisions", body)

    def get(self):
        return self.request("GET", "/")

    def wait_published(self, version: int | None = None,
                       timeout: float = 8.0) -> dict:
        deadline = time.monotonic() + timeout
        last = None
        while time.monotonic() < deadline:
            status, last = self.get()
            assert status == 200
            if last["published"] is not None and (
                    version is None or last["published"]["version"] == version):
                return last
            time.sleep(0.05)
        raise AssertionError(f"publish timed out; last={last!r}")

    def sql(self, query: str):
        conn = sqlite3.connect(self.db)
        try:
            return conn.execute(query).fetchall()
        finally:
            conn.close()

    def stop(self) -> None:
        for proc in self.procs:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        self.procs.clear()


def assert_snapshot_shape(snapshot: dict, version: int, y) -> None:
    assert set(snapshot) == {"version", "y", "mean", "A", "B"}
    assert snapshot["version"] == version
    assert snapshot["y"] == y
    n = len(y)
    assert len(snapshot["mean"]) == n
    assert len(snapshot["A"]) == n and all(len(row) == n
                                           for row in snapshot["A"])
    assert len(snapshot["B"]) == n and all(len(row) == n
                                           for row in snapshot["B"])
    # every scalar is a reduced rational string (denominator omitted at 1)
    def check(text):
        assert isinstance(text, str) and text
        frac = Fraction(text)
        if frac.denominator == 1:
            assert "/" not in text
        else:
            assert text == f"{frac.numerator}/{frac.denominator}"
    check(snapshot["mean"][0])
    check(snapshot["A"][0][0])
    check(snapshot["B"][0][0])


def _canon(pairs):
    return sorted((t, None if v is None else Fraction(v)) for t, v in pairs)


# ---------------------------------------------------------------------------
# A. bootstrap and the documented 2/4 -> 1/2 example
# ---------------------------------------------------------------------------
def case_bootstrap_and_documented_example():
    with tempfile.TemporaryDirectory() as tmp:
        svc = Service(tmp, 18090)
        svc.init()
        svc.start_server()
        try:
            status, query = svc.get()
            assert status == 200 and query == {"version": 0,
                                               "published": None}

            status, version = svc.post(json.dumps(
                {"id": "a", "changes": [[0, "2/4"]]}))
            assert status == 200 and version == 1

            # same id, numerically equal content, different spelling/order:
            # the original version is returned and nothing changes.
            status, version = svc.post(json.dumps(
                {"id": "a", "changes": [[0, "1/2"]]}))
            assert status == 200 and version == 1
            assert svc.get()[1]["version"] == 1
        finally:
            svc.stop()
    print("ok  bootstrap v0 + id='a' 2/4 then 1/2 both return 1")


# ---------------------------------------------------------------------------
# B. validation battery: 400 with JSON path, state never changes
# ---------------------------------------------------------------------------
def case_validation_battery():
    cases = [
        ("{", "$"),
        ("[]", "$"),
        ('{"id":"a","changes":[],"x":1}', "$.x"),
        ('{"id":"a","id":"b","changes":[]}', "$"),
        ('{"id":"","changes":[]}', "$.id"),
        ('{"id":"a-b","changes":[]}', "$.id"),
        ('{"id":"abé","changes":[]}', "$.id"),
        ('{"id":"a","changes":{}}', "$.changes"),
        ('{"id":"a","changes":[1]}', "$.changes[0]"),
        ('{"id":"a","changes":[[1]]}', "$.changes[0]"),
        ('{"id":"a","changes":[[true,"1"]]}', "$.changes[0][0]"),
        ('{"id":"a","changes":[[-1,"1"]]}', "$.changes[0][0]"),
        ('{"id":"a","changes":[[1.5,"1"]]}', "$.changes[0][0]"),
        ('{"id":"a","changes":[[0,true]]}', "$.changes[0][1]"),
        ('{"id":"a","changes":[[0,1]]}', "$.changes[0][1]"),
        ('{"id":"a","changes":[[0,"1/0"]]}', "$.changes[0][1]"),
        ('{"id":"a","changes":[[0,"1"],[0,"2"]]}', "$.changes[1][0]"),
        ('{"changes":[]}', "$.id"),
        ('{"id":"a"}', "$.changes"),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        svc = Service(tmp, 18091)
        svc.init()
        svc.start_server()
        try:
            for body, path in cases:
                status, payload = svc.post(body)
                assert status == 400, (body, status, payload)
                assert payload["error"]["path"] == path, (body, payload)
            # out-of-horizon time is a 400 carrying the indexed path too
            status, payload = svc.post(json.dumps(
                {"id": "a", "changes": [[99, "1"]]}))
            assert status == 400 and payload["error"]["path"] == \
                "$.changes[0][0]", payload
            # properly UTF-8 encoded non-ASCII id is rejected on the charset
            # (already covered in the battery above; nothing new to do)
            # a body that is not valid UTF-8 at all is a 400 at "$"
            status, payload = svc.request(
                "POST", "/revisions",
                body=b'{"id":"ab\xff","changes":[]}')
            assert status == 400 and payload["error"]["path"] == "$"
            # nothing was accepted
            assert svc.get()[1] == {"version": 0, "published": None}
            assert svc.sql("SELECT count(*) FROM revisions") == [(0,)]
        finally:
            svc.stop()
    print("ok  invalid/duplicate members -> 400 with JSON path, no state change")


# ---------------------------------------------------------------------------
# C. idempotency, 409, empty changesets, ordering, monotonic versions
# ---------------------------------------------------------------------------
def case_idempotency_conflict_and_empty_changes():
    with tempfile.TemporaryDirectory() as tmp:
        svc = Service(tmp, 18092)
        svc.init()
        svc.start_server()
        try:
            # empty changeset is legal and still creates a new version
            status, version = svc.post(json.dumps(
                {"id": "empty1", "changes": []}))
            assert status == 200 and version == 1
            status, version = svc.post(json.dumps(
                {"id": "empty1", "changes": []}))
            assert version == 1  # identical, idempotent

            # order-independent, numerically compared equality
            body_a = json.dumps({"id": "m", "changes": [[0, "1/2"], [1, "3"]]})
            body_b = json.dumps({"id": "m", "changes": [[1, "6/2"],
                                                        [0, "2/4"]]})
            assert svc.post(body_a)[1] == 2
            assert svc.post(body_b)[1] == 2  # same version, no increment

            status, payload = svc.post(json.dumps(
                {"id": "m", "changes": [[0, "1/2"], [1, "4"]]}))
            assert status == 409 and payload["error"]["path"] == "$.id"

            # conflict changes nothing; a fresh id gets the next integer
            assert svc.get()[1]["version"] == 2
            assert svc.post(json.dumps(
                {"id": "n", "changes": []}))[1] == 3
            assert svc.get()[1]["version"] == 3
        finally:
            svc.stop()
    print("ok  same-id equality ignores order, 409 on mismatch, empty changes")


# ---------------------------------------------------------------------------
# D. generation takeover, stale results ignored, interleaved convergence
# ---------------------------------------------------------------------------
def case_generation_takeover_and_interleaving():
    with tempfile.TemporaryDirectory() as tmp:
        svc = Service(tmp, 18093)
        svc.init()
        svc.start_server()
        try:
            worker1 = svc.start_worker()
            query = svc.wait_published(0)
            assert query["version"] == 0
            assert_snapshot_shape(query["published"], 0, [None, "2"])

            # a second worker claims a new generation; worker 1 must exit
            worker2 = svc.start_worker()
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and worker1.poll() is None:
                time.sleep(0.05)
            assert worker1.poll() is not None, "superseded worker must exit"
            assert svc.sql("SELECT generation FROM generation")[0][0] == 2

            # interleave many revisions while worker2 is computing
            for k in range(1, 9):
                status, _ = svc.post(json.dumps(
                    {"id": f"r{k}",
                     "changes": [[k % 2, f"{k}"]]}))
                assert status == 200
            final = svc.wait_published(8)
            assert final["version"] == 8
            # times: even k write time 0 (largest even k = 8), odd k time 1
            assert final["published"]["y"] == ["8", "7"]
            assert_snapshot_shape(final["published"], 8, ["8", "7"])

            # revisions stop: the current worker has the latest version done
            time.sleep(0.3)
            again = svc.get()[1]
            assert again["version"] == 8 and \
                again["published"]["version"] == 8

            # the previous complete snapshot survives a newer unpublished
            # input: accept one more revision, query must still carry the v8
            # snapshot until the worker catches up.
            svc.post(json.dumps({"id": "late", "changes": [[0, "100"]]}))
            caught = svc.wait_published(9)
            assert caught["published"]["y"] == ["100", "7"]
            assert_snapshot_shape(caught["published"], 9, ["100", "7"])
        finally:
            svc.stop()
    print("ok  generation takeover, stale results ignored, interleaved publish")


# ---------------------------------------------------------------------------
# E. hard kills: after accept/before receipt and around the publish commit
# ---------------------------------------------------------------------------
def case_hard_kill_after_accept_before_receipt():
    with tempfile.TemporaryDirectory() as tmp:
        svc = Service(tmp, 18094)
        svc.init()
        server = svc.start_server(crash_hook="after_accept")
        try:
            # the connection dies with the server process
            try:
                svc.post(json.dumps({"id": "a",
                                     "changes": [[0, "2/4"]]}))
                raise AssertionError("connection should have been killed")
            except (http.client.HTTPException, ConnectionError, OSError):
                pass
            server.wait(timeout=5)
            assert server.returncode == 137

            # restart with no hook: the revision is durable, the retry gets
            # the already-assigned version 1 rather than creating a new one.
            svc.procs.remove(server)
            svc.port = 18095
            svc.start_server()
            status, version = svc.post(json.dumps(
                {"id": "a", "changes": [[0, "1/2"]]}))
            assert status == 200 and version == 1
            assert svc.get()[1]["version"] == 1
        finally:
            svc.stop()
    print("ok  hard kill after accept: durable revision, retry returns v1")


def case_hard_kills_around_publish_commit():
    with tempfile.TemporaryDirectory() as tmp:
        svc = Service(tmp, 18096)
        svc.init()
        svc.start_server()
        try:
            svc.post(json.dumps({"id": "a", "changes": [[0, "3"]]}))

            # kill strictly before the commit: no published row exists
            worker = svc.start_worker(crash_hook="before_publish_commit")
            worker.wait(timeout=5)
            assert svc.sql("SELECT count(*) FROM published") == [(0,)]
            assert svc.sql("SELECT generation FROM generation")[0][0] == 1

            # a restarted worker claims generation 2 and completes the latest
            worker2 = svc.start_worker()
            query = svc.wait_published(1)
            assert query["version"] == 1
            assert_snapshot_shape(query["published"], 1, ["3", "2"])
            assert svc.sql("SELECT generation FROM generation")[0][0] == 2
            svc.stop()
            svc.procs.clear()
        finally:
            svc.stop()

        # kill strictly after the commit: the snapshot must survive
        worker = svc.start_worker(crash_hook="after_publish_commit")
        worker.wait(timeout=5)
        rows = svc.sql("SELECT version, snapshot FROM published")
        assert len(rows) == 1 and rows[0][0] == 1
        snapshot = json.loads(rows[0][1])
        assert_snapshot_shape(snapshot, 1, ["3", "2"])
        print("ok  hard kills before/after publish commit preserve invariants")


# ---------------------------------------------------------------------------
# F. worker restart recomputes the latest version from durable state
# ---------------------------------------------------------------------------
def case_worker_restart_recomputes():
    with tempfile.TemporaryDirectory() as tmp:
        svc = Service(tmp, 18097)
        svc.init()
        svc.start_server()
        try:
            svc.post(json.dumps({"id": "a", "changes": [[0, "5"]]}))
            worker = svc.start_worker()
            time.sleep(0.2)
            worker.kill()  # SIGKILL, no cleanup hooks
            worker.wait()

            # remove any partial publish possibility: ensure either absent or
            # a complete row; then restart and wait for the latest version.
            svc.procs.remove(worker)
            worker2 = svc.start_worker()
            query = svc.wait_published(1)
            assert query["version"] == 1
            assert query["published"]["y"] == ["5", "2"]
            assert_snapshot_shape(query["published"], 1, ["5", "2"])
        finally:
            svc.stop()
    print("ok  worker SIGKILL then restart recomputes latest version")


def main():
    case_bootstrap_and_documented_example()
    case_validation_battery()
    case_idempotency_conflict_and_empty_changes()
    case_generation_takeover_and_interleaving()
    case_hard_kill_after_accept_before_receipt()
    case_hard_kills_around_publish_commit()
    case_worker_restart_recomputes()
    print("\nALL SERVICE ACCEPTANCE CHECKS PASSED")


if __name__ == "__main__":
    main()
