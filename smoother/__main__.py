"""Command line entry: ``python -m smoother INPUT.json OUTPUT.json``.

The output file is replaced only after a fully successful run: the result is
written to a temporary file in the same directory and atomically renamed, so
any validation or computation failure leaves a pre-existing output file
untouched.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

from .core import solve, to_jsonable
from .validate import ValidationError, load_input


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        sys.stderr.write("usage: python -m smoother INPUT.json OUTPUT.json\n")
        return 2
    in_path, out_path = argv

    try:
        with open(in_path, "r", encoding="utf-8") as fh:
            text = fh.read()
        inp = load_input(text)
        result = to_jsonable(solve(inp))
    except ValidationError as exc:
        sys.stderr.write(json.dumps(exc.payload(), ensure_ascii=False) + "\n")
        return 1
    except (OSError, UnicodeDecodeError) as exc:
        sys.stderr.write(json.dumps(
            {"error": {"path": "$", "message": f"cannot read input: {exc}"}},
            ensure_ascii=False) + "\n")
        return 1

    out_dir = os.path.dirname(os.path.abspath(out_path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".smoother-", suffix=".tmp",
                                    dir=out_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, separators=(",", ":"))
            fh.write("\n")
        os.replace(tmp_path, out_path)
    except OSError as exc:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        sys.stderr.write(json.dumps(
            {"error": {"path": "$", "message": f"cannot write output: {exc}"}},
            ensure_ascii=False) + "\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
