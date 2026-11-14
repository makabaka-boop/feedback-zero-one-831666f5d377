"""Persistent HTTP revision service for the limiting smoother.

The package adds a durable, SQLite-backed front end and an independent worker
process around the exact rational smoother:

* :mod:`smoother.service.schema`    -- SQLite schema, model bootstrap, queries
* :mod:`smoother.service.revisions` -- strict validation of revision requests
* :mod:`smoother.service.server`    -- the HTTP front end (acceptance only)
* :mod:`smoother.service.worker`    -- the independent computing process

The original one-shot CLI (``python -m smoother`` / ``run_smoother.py``) is
unchanged.
"""
