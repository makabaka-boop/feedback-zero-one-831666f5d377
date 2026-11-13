#!/usr/bin/env python3
"""Thin entry point: python run_smoother.py INPUT.json OUTPUT.json."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from smoother.__main__ import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
