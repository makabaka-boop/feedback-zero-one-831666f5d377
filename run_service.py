#!/usr/bin/env python3
"""Top-level entry for the persistent HTTP service.

Examples
--------
    python3 -I run_service.py init   data.db model.json
    python3 -I run_service.py serve  data.db --port 8080
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from smoother.service.__main__ import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
