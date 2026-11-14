"""Deterministic crash points for acceptance testing of hard-kill recovery.

The acceptance suite must be able to kill a process at precise moments
(after a revision has been accepted but before its receipt is returned, and
before/after the publish transaction commits).  The hooks are inert unless the
``SMOOTHER_CRASH_HOOK`` environment variable names a point, so production runs
never pay for them beyond an environment lookup.
"""

from __future__ import annotations

import os

#: After the revision row is committed, before the HTTP response is sent.
AFTER_ACCEPT = "after_accept"
#: Inside the worker, immediately before the publish transaction commits.
BEFORE_PUBLISH_COMMIT = "before_publish_commit"
#: Inside the worker, immediately after the publish transaction commits.
AFTER_PUBLISH_COMMIT = "after_publish_commit"

CRASH_HOOK_ENV = "SMOOTHER_CRASH_HOOK"


def crash_point(name: str) -> None:
    """Terminate this process with SIGKILL when the named hook is armed."""
    if os.environ.get(CRASH_HOOK_ENV) == name:
        # Remove the hook first so a restarted process in the same
        # environment does not instantly die again; the test harness only
        # arms it for the process it wants killed.
        os.environ.pop(CRASH_HOOK_ENV, None)
        os._exit(137)
