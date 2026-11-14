"""Persistent HTTP front end for revision acceptance.

The server only *accepts* revisions: it validates each request and commits it
to SQLite.  It never runs the smoother itself, so computation in the worker
process cannot block a revision.  Two routes are served:

``POST /revisions``
    body: a revision document; 200 with an integer version on success,
    400 with a JSON-path error payload for invalid input, 409 for an id that
    already exists with different content.

``GET /``
    atomically returns ``{"version": <input version>,
    "published": <snapshot or null>}``.

Every request opens its own SQLite connection; writes use
``BEGIN IMMEDIATE`` and therefore serialize safely across threads.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ..validate import ValidationError
from . import schema
from .revisions import parse_revision


class RevisionHandler(BaseHTTPRequestHandler):
    server_version = "SmootherService/1.0"
    protocol_version = "HTTP/1.0"
    db_path: str = ""  # set by :func:`serve` on the handler class

    # ------------------------------------------------------------------
    # response helpers
    # ------------------------------------------------------------------
    def _send_json(self, status: int, payload) -> None:
        body = json.dumps(payload, ensure_ascii=False,
                          separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: int, error: ValidationError) -> None:
        self._send_json(status, error.payload())

    def log_message(self, fmt, *args) -> None:  # quiet by default
        if getattr(self.server, "verbose", False):
            super().log_message(fmt, *args)

    # ------------------------------------------------------------------
    # routes
    # ------------------------------------------------------------------
    def do_GET(self) -> None:
        if self.path != "/":
            self._send_json(404, {"error": {
                "path": "$",
                "message": f"unknown route {self.path!r}; use GET /"}})
            return
        conn = schema.connect(self.db_path)
        try:
            self._send_json(200, schema.get_query(conn))
        finally:
            conn.close()

    def do_POST(self) -> None:
        if self.path != "/revisions":
            self._send_json(404, {"error": {
                "path": "$",
                "message": "unknown route; use POST /revisions"}})
            return

        length = self.headers.get("Content-Length")
        try:
            size = int(length) if length is not None else -1
            if size < 0:
                raise ValueError
        except ValueError:
            self._send_json(400, ValidationError(
                "$", "missing or invalid Content-Length header").payload())
            return
        raw = self.rfile.read(size)
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            self._send_json(400, ValidationError(
                "$", "request body must be UTF-8 JSON").payload())
            return

        # All parsing/validation happens before any state is touched, so a
        # rejected request cannot change the database.
        try:
            rev_id, changes = parse_revision(text)
        except ValidationError as exc:
            self._send_error(400, exc)
            return

        conn = schema.connect(self.db_path)
        try:
            try:
                version = schema.submit_revision(conn, rev_id, changes)
            except ValidationError as exc:
                self._send_error(400, exc)
                return
            except schema.ConflictError as exc:
                self._send_json(409, exc.payload())
                return
        finally:
            conn.close()
        self._send_json(200, version)


def serve(db_path: str, host: str, port: int,
          verbose: bool = False) -> ThreadingHTTPServer:
    """Create and return (without starting) the HTTP server."""
    handler = type("BoundRevisionHandler", (RevisionHandler,),
                   {"db_path": db_path})
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.verbose = verbose
    return httpd
