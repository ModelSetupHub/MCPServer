"""Frontend layer for the ModelSetupHub MCP server.

Everything to do with the in-chat progress panel, kept out of ``main.py``:

- ``assets/`` — the panel's HTML, CSS and JavaScript, one file each.
- ``loader`` — inlines those assets into the single HTML document an MCP Apps
  ``ui://`` resource has to be.
- ``jobs`` — the ``Job`` model, the registry of jobs running now, and the
  persistence that keeps every job's snapshot on disk and its benchmark result
  in the benchmark history.
- ``workers`` — the threads that run the operations and keep their jobs current,
  fed by MSHCore's own progress reporting: the download manager's status for
  downloads, the benchmark runner's ``on_progress`` callback for benchmarks.
- ``app`` — the Apps extension: the ``ui://`` resource plus the tools bound to it,
  and ``register_progress_tools`` for the plain tools the model polls with.

The shape of the system:

    a tracked tool  → creates a Job, persists it, starts a worker, returns its id
    the worker      → updates the Job, persists snapshots, finishes it exactly once
    the panel       → polls its own app-visible status tool with that id and draws
                      the result
    the model       → polls the plain progress_get_status with the same id

Only the two tools that start an operation are bound to the panel, because a
client renders one panel per tool result: binding the poll would draw a second bar
for every poll of the first.

``main.py`` imports ``create_progress_app`` from here and passes the result to
``MCPServer(extensions=[...])``, and calls ``register_progress_tools`` alongside
its own registrars.
"""

from .app import (
    PROGRESS_URI,
    create_progress_app,
    register_progress_tools,
)
from .workers import (
    note_download_ended,
    start_benchmark,
    start_download,
)

__all__ = [
    "PROGRESS_URI",
    "create_progress_app",
    "register_progress_tools",
    "note_download_ended",
    "start_benchmark",
    "start_download",
]
