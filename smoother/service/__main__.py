"""Command line entry for the persistent service.

Subcommands
-----------
``init DB MODEL.json``
    create ``DB`` and bootstrap version 0 from a legal model file; refuses an
    already-initialized database.

``serve DB [--host H] [--port P]``
    run the SQLite-backed HTTP revision server (the schema must already
    exist, created by ``init``).

``worker DB [--poll SECONDS]``
    run one independent worker process: it claims a fresh generation token
    on startup and publishes only current-generation, current-version
    results.

The one-shot ``python -m smoother INPUT OUTPUT`` entry is unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys

from ..validate import ValidationError
from . import schema
from .server import serve
from .worker import run_worker


def _cmd_init(args: argparse.Namespace) -> int:
    try:
        with open(args.model, "r", encoding="utf-8") as fh:
            text = fh.read()
        conn = schema.connect(args.database)
        try:
            schema.initialize(conn, text)
        finally:
            conn.close()
    except ValidationError as exc:
        sys.stderr.write(json.dumps(exc.payload(), ensure_ascii=False) + "\n")
        return 1
    except (OSError, UnicodeDecodeError) as exc:
        sys.stderr.write(json.dumps(
            {"error": {"path": "$", "message": f"cannot read model: {exc}"}},
            ensure_ascii=False) + "\n")
        return 1
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    conn = schema.connect(args.database)
    try:
        initialized = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='meta'"
        ).fetchone()
        if initialized is None:
            sys.stderr.write(json.dumps({
                "error": {"path": "$",
                          "message": "database is not initialized; run init"}},
                ensure_ascii=False) + "\n")
            return 1
    finally:
        conn.close()

    httpd = serve(args.database, args.host, args.port, verbose=args.verbose)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


def _cmd_worker(args: argparse.Namespace) -> int:
    run_worker(args.database, poll_interval=args.poll)
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(
        prog="python -m smoother.service",
        description="Persistent HTTP revision service for the smoother.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="initialize a database from a model")
    p_init.add_argument("database")
    p_init.add_argument("model")
    p_init.set_defaults(func=_cmd_init)

    p_serve = sub.add_parser("serve", help="run the HTTP server")
    p_serve.add_argument("database")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8080)
    p_serve.add_argument("--verbose", action="store_true")
    p_serve.set_defaults(func=_cmd_serve)

    p_worker = sub.add_parser("worker", help="run a worker process")
    p_worker.add_argument("database")
    p_worker.add_argument("--poll", type=float, default=0.1)
    p_worker.set_defaults(func=_cmd_worker)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
