#!/usr/bin/env python3
"""Top-level entry for an independent worker process.

    python3 -I run_worker.py data.db [--poll SECONDS]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from smoother.service.worker import run_worker  # noqa: E402


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    poll = 0.1
    positional = []
    while argv:
        token = argv.pop(0)
        if token == "--poll":
            poll = float(argv.pop(0))
        else:
            positional.append(token)
    if len(positional) != 1:
        sys.stderr.write("usage: python run_worker.py DATABASE [--poll S]\n")
        return 2
    run_worker(positional[0], poll_interval=poll)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
