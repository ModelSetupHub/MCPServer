"""The progress panel: its MCP Apps extension, and the tools that read it.

MCP Apps (``io.modelcontextprotocol/ui``) is how a tool result carries a UI: the
tool advertises ``_meta.ui.resourceUri``, which points at a ``ui://`` HTML
resource the client renders inline in the conversation, in a sandboxed iframe.

That binding is what draws a progress bar, so it decides which tools may have it.
A tool bound to the panel gets a new panel every time the *model* calls it, since
each call is a new tool result in the conversation for the host to render. So only
the tools that start an operation are bound to it — one call, one job, one
bar — and everything that merely *reads* or *controls* an existing job is a plain
tool, registered by :func:`register_progress_tools`, with no UI of its own.

Two surfaces, therefore, over one implementation:

- The model calls ``progress_get_status``, ``benchmark_get_result``,
  ``progress_cancel`` and ``progress_pause``. None is bound to the panel, so
  polling a benchmark to completion adds no second bar to the conversation.
- The panel calls ``progress_panel_status``, ``progress_panel_cancel`` and
  ``progress_panel_pause`` over its postMessage bridge. These are bound to the
  panel and marked ``visibility=["app"]``, so they serve the view it belongs to
  without appearing in the model's tool list. A call the panel makes itself is
  answered back to the iframe rather than added to the conversation, which is why
  polling twice a second never renders anything.

Both surfaces call the same four functions below, so the model and the view can
never disagree about a job.

One identifier, one lookup. A status request resolves the exact id it is given:
the running job when there is one, its persisted record otherwise. There is no
search, no fallback and no guessing, so a request for one run can never be
answered with another. Live and reopened panels take the same path, because there
is only one.

Every tool degrades to plain text: a client that did not negotiate Apps receives
the same return value it would from the corresponding tool in ``main.py``.
"""

from __future__ import annotations

from collections.abc import Callable
import functools
from pathlib import Path
from typing import Any, TypeVar
from uuid import uuid4

from mcp.server.apps import Apps
from mcp.server.mcpserver.exceptions import MCPServerError, ToolError
from mcp.types import ToolAnnotations

from MSHCore.download_manager.manager import DownloadManager
from MSHCore.paths import DOWNLOADS_DIRECTORY

from . import workers
from .jobs import Job, load_job_result, load_snapshot, registry
from .loader import load_progress_app_html

PROGRESS_URI = "ui://modelsetuphub/progress.html"

LONG_RUNNING = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=False,
    open_world_hint=True,
)

CallableT = TypeVar("CallableT", bound=Callable[..., Any])


def surface_core_errors(function: CallableT) -> CallableT:
    """Forward exceptions raised by MSHCore to the MCP client verbatim.

    Mirrors the decorator in ``main.py``: the SDK reports any exception other than
    ``ToolError`` as a generic tool crash, which would hide the descriptive
    messages MSHCore raises.

    Args:
        function: Tool function that calls into the MSHCore package.

    Returns:
        CallableT: Wrapped function preserving the original error text.
    """

    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return function(*args, **kwargs)
        except MCPServerError:
            raise
        except Exception as error:
            raise ToolError(f"{type(error).__name__}: {error}") from error

    return wrapper  # type: ignore[return-value]


def create_progress_app(
    get_session: Callable[[str], Any],
    release_session: Callable[[str], Any] | None = None,
    register_session: Callable[[str, Any], None] | None = None,
) -> Apps:
    """Build the Apps extension carrying the panel and the tools that draw it.

    Only the tools that *start* an operation are bound to the panel, plus the
    panel's own app-visible ones. Reading and controlling a job that already exists
    is registered separately by :func:`register_progress_tools`, so the model can
    poll a benchmark without every poll rendering another progress bar.

    Args:
        get_session: Resolver for a download session id, supplied by the layer
            that owns the session registry — ``main.py`` — so this module does not
            duplicate that state.
        release_session: Called with a session id once its queue has stopped, so
            the owning layer can drop a cancelled session; the single-file
            download tool also calls it when its own setup fails.
        register_session: Adds a freshly created session to the owning layer's
            registry. The single-file download tool generates its own session
            id and needs it placed where the plain download tools find it.

    Returns:
        Apps: Extension to pass as ``MCPServer(extensions=[...])``.
    """
    workers.release_session = release_session

    apps = Apps()

    apps.add_html_resource(
        PROGRESS_URI,
        load_progress_app_html(),
        name="progress-panel",
        title="Operation progress",
        description=(
            "Live progress for downloads and benchmarks, rendered inline in the "
            "conversation."
        ),
        prefers_border=False,
    )

    _register_panel_tools(apps)
    _register_download(apps, get_session, register_session, release_session)
    _register_benchmarks(apps)
    _register_models(apps)

    return apps


def register_progress_tools(server: Any) -> None:
    """Register the tools that read and control an operation already running.

    Deliberately plain tools, on the server rather than the Apps extension. A tool
    bound to the panel renders one every time the model calls it, and the model
    calls these repeatedly by design — a benchmark is polled until it finishes.
    Binding them would turn one benchmark into a row of progress bars, each drawn
    from a poll's answer rather than from the run itself.

    They return exactly what the panel's own tools return, because both call the
    same four functions.

    Args:
        server: Server instance the tools are attached to, matching the registrars
            in ``main.py``.
    """

    @server.tool(
        name="progress_get_status",
        title="Get operation progress",
        description=(
            "Report one asynchronous operation's current state, by the progress_id "
            "its tracked tool returned. Fast and non-blocking: it reads the "
            "operation's recorded state, it does not wait for it. Returns one "
            "dict with exactly these keys: 'found' (true), 'id', 'type' "
            "('download', 'benchmark' or 'addmodel'), 'title', 'status' "
            "('starting' or 'running' while the work continues, then one of "
            "'completed', 'failed' or 'cancelled'), 'progress' (0-100 rounded "
            "to one decimal, or null when the remaining work cannot be "
            "measured), 'message', 'error' (set on a failure), "
            "'result_available' (a completed benchmark's measurements are "
            "ready to fetch with benchmark_get_result; the measurements "
            "themselves are never included here, and this is always false "
            "for downloads and model imports), 'benchmark_id' (the history "
            "id a completed benchmark was saved under, null otherwise), "
            "'steps' (one row per downloaded file, or per "
            "model-configuration pair, or per prompt for a single-pair "
            "benchmark; an import carries no steps and no status text — "
            "its view is the title alone; each with 'name', 'state' — one of "
            "'waiting', 'running', 'completed', 'failed', 'skipped', "
            "'cancelled' — 'percent', 'detail' and 'error'), 'metrics' "
            "(label/value pairs: downloads report downloaded and total "
            "bytes, file n of m, and the current speed while something is "
            "transferring; benchmarks and model imports report none), "
            "'paused', 'cancelling' "
            "(true between a cancel request and the operation actually "
            "stopping), 'can_cancel', 'can_pause' (downloads only) and "
            "'elapsed_seconds'. Do not poll in a held-open "
            "loop: after starting an operation, end the turn and go to sleep, "
            "then call this when the user next prompts, repeating only if the "
            "status is not yet terminal — the operation's own progress bar keeps "
            "updating itself either way. Answers while the operation runs and "
            "afterwards, including after this server restarts. An unknown id "
            "returns only 'id', 'found' (false) and an explanatory 'message', "
            "never another operation's progress."
        ),
        annotations=ToolAnnotations(
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=False,
            open_world_hint=False,
        ),
    )
    @surface_core_errors
    def progress_get_status(progress_id: str) -> dict:
        """Return one job's snapshot.

        Args:
            progress_id: Identifier returned by a tracked tool.

        Returns:
            dict: The snapshot, or a not-found response.
        """
        return read_status(progress_id)

    @server.tool(
        name="benchmark_get_result",
        title="Get a finished benchmark's measurements",
        description=(
            "Retrieve the measurements produced by a benchmark started with "
            "one of the four benchmark_ollama_* tools (single, configs, "
            "models or tournament), "
            "by its progress_id. This is where a benchmark's actual results come "
            "from: the starting tool returns only a handle, and progress_get_status "
            "reports only progress. Call this once progress_get_status reports "
            "status='completed' and result_available=true — earlier it reports that "
            "the benchmark is still running, and for a failed or cancelled run it "
            "reports that there are no measurements. Returns one dict with "
            "'id', 'found', 'status', 'result_available' and, when the "
            "measurements exist, 'result'. For a run with a single "
            "model-configuration pair the result is that pair's own "
            "measurements: 'model', 'name' (the configuration's label), "
            "'configuration' (the options it ran under), 'repetitions', "
            "'results' (one entry per prompt, averaged over the repetitions: "
            "'prompt', 'success', 'duration_seconds', 'prompt_tokens', "
            "'output_tokens', 'prompt_tokens_per_second', "
            "'output_tokens_per_second', 'ttft_seconds', and with an NVIDIA "
            "GPU 'vram_used_mb', 'gpu_temperature_c' and 'gpu_clock_mhz' — "
            "each null when that figure was not measured, each timing and "
            "rate also carrying '_stddev', '_min' and '_max' over the "
            "repetitions, a single repetition reporting 0.0 and its own "
            "value, and 'response' holding the generated text when the run "
            "was started with include_output=true; a prompt whose every "
            "repetition failed reports success=false with its 'error' "
            "instead of numbers) and 'summary' ('average_duration_seconds', "
            "'average_prompt_tokens_per_second', "
            "'average_output_tokens_per_second', 'total_output_tokens' and "
            "'output_tokens_per_second_stddev' — the run-to-run noise level, "
            "null without repetitions). For several pairs the result is the "
            "whole comparison: 'experiments' (the normalized matrix that "
            "ran), 'models' in run order, 'tests' holding one of the above "
            "per pair under its label, and 'significance' with 'by_model' "
            "(each model's configurations judged against each other) and "
            "'across_models' (the models judged against each other, each "
            "standing in under its model's name) — each verdict carrying "
            "'metric', 'leader', 'runner_up', 'difference', 'significant' "
            "(true/false when repetitions ran above 1, null when each "
            "prompt was measured once) and 'message'. A failed or "
            "cancelled run answers found=true with result_available=false "
            "and the reason why there is nothing to fetch."
        ),
        annotations=ToolAnnotations(
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        ),
    )
    @surface_core_errors
    def benchmark_get_result(progress_id: str) -> dict:
        """Return a finished benchmark's measurements.

        Args:
            progress_id: Identifier returned by a benchmark tool.

        Returns:
            dict: The measurements, or an explanation of why there are none yet.
        """
        return read_result(progress_id)

    @server.tool(
        name="progress_cancel",
        title="Cancel a running operation",
        description=CANCEL_DESCRIPTION,
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=True,
            idempotent_hint=True,
            open_world_hint=False,
        ),
    )
    @surface_core_errors
    def progress_cancel(progress_id: str) -> dict:
        """Cancel one operation and wait for its cleanup.

        Args:
            progress_id: Identifier returned by a tracked tool.

        Returns:
            dict: Final snapshot after the cleanup.
        """
        return cancel_operation(progress_id)

    @server.tool(
        name="progress_pause",
        title="Stop or resume a download",
        description=PAUSE_DESCRIPTION,
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=False,
            open_world_hint=False,
        ),
    )
    @surface_core_errors
    def progress_pause(progress_id: str) -> dict:
        """Suspend or continue one download.

        Args:
            progress_id: Identifier returned by a tracked tool.

        Returns:
            dict: Snapshot afterwards, with ``pause_action``.
        """
        return pause_operation(progress_id)


def _snapshot(progress_id: str) -> dict | None:
    """Resolve one job's current snapshot.

    The single read path, used by the panel whether it is following a live
    operation or showing one after the conversation was reopened. The running job
    answers when there is one, because its record trails it by up to half a second;
    otherwise the record does, which is why a finished run survives a restart.

    Args:
        progress_id: Identifier returned by a tracked tool.

    Returns:
        dict | None: The snapshot, or None when no such job exists.
    """
    job = registry.get(progress_id)

    return job.snapshot() if job is not None else load_snapshot(progress_id)


def _not_found(progress_id: str | None) -> dict:
    """Build the answer for an identifier that names no job.

    ``found`` is false and there is no status: a missing job has no lifecycle
    state, and reporting one would let the panel treat a bad id — or a record
    pruned months later — as an operation that ended.

    Args:
        progress_id: Identifier that was asked for.

    Returns:
        dict: Not-found response.
    """
    return {
        "id": progress_id,
        "found": False,
        "message": (
            "No operation is recorded under that progress_id. It may have been "
            "mistyped, or its record may have been pruned."
        ),
    }


def read_status(progress_id: str) -> dict:
    """Report one operation's current state.

    The one implementation behind both status tools: the model's plain
    ``progress_get_status`` and the panel's ``progress_panel_status``. Polling it
    is a read and nothing else — no job is created, none is touched, and the answer
    describes only the id it was given.

    Args:
        progress_id: Identifier returned by a tracked tool.

    Returns:
        dict: The snapshot with ``found`` true, or a not-found response.
    """
    snapshot = _snapshot(progress_id)

    if snapshot is None:
        return _not_found(progress_id)

    return {**snapshot, "found": True}


def read_result(progress_id: str) -> dict:
    """Hand over a finished benchmark's measurements.

    Args:
        progress_id: Identifier returned by a benchmark tool.

    Returns:
        dict: The measurements, or an explanation of why there are none yet.
    """
    snapshot = _snapshot(progress_id)

    if snapshot is None:
        return _not_found(progress_id)

    job = registry.get(progress_id)
    result = job.result() if job is not None else load_job_result(progress_id)

    if result is None:
        return {
            "id": progress_id,
            "found": True,
            "status": snapshot.get("status"),
            "result_available": False,
            "message": _no_result_reason(snapshot),
        }

    # The history stores every run as a comparison; a single test is unwrapped
    # to that test's own measurements, here rather than at write time.
    result = workers._business_result(result)

    return {
        "id": progress_id,
        "found": True,
        "status": snapshot.get("status"),
        "result_available": True,
        "result": result,
    }


def _no_result_reason(snapshot: dict) -> str:
    """Explain why a job has no measurements to hand over.

    Args:
        snapshot: The job's current snapshot.

    Returns:
        str: The reason, phrased for the model.
    """
    status = snapshot.get("status")

    if status in ("starting", "running"):
        return (
            "This benchmark is still running. End the turn and go to sleep; when "
            "the user calls again, poll progress_get_status with this "
            "progress_id until its status is completed, then call this again."
        )

    if status == "failed":
        return (
            f"This benchmark failed, so it produced no measurements: "
            f"{snapshot.get('error') or 'no error was recorded'}."
        )

    if status == "cancelled":
        return "This benchmark was cancelled, so it produced no measurements."

    return "No measurements were recorded for this operation."


CANCEL_DESCRIPTION = (
    "Cancel the operation with this progress_id and undo what it had done. "
    "The download, benchmark or model import stops at its next safe point — "
    "a chunk boundary for a download, the next process kill for an import — "
    "and MSHCore undoes what it created: a download's partial and completed "
    "files are deleted, with files that existed before it untouched, a "
    "benchmark's loaded model is unloaded and its partial measurements "
    "discarded, and an import's 'ollama create' process is killed so nothing "
    "is registered under the model's name. Every kind records a 'cancelled' "
    "entry in the execution log where logs_read will show it. For a "
    "download the session is removed too, so every download_* tool refuses "
    "its session_id afterwards and downloading the same files again means "
    "creating it fresh. Cannot be undone; to suspend a download and keep "
    "it, use progress_pause. Blocks up to 60 seconds for the cleanup. "
    "Returns the final snapshot, in the same shape as progress_get_status, "
    "plus 'cancel_requested' (true when this call requested the "
    "cancellation) and 'cleanup_complete' (false only when MSHCore was "
    "still cleaning up when the wait ended — the operation will not resume "
    "either way). For an unknown id or one that already finished, nothing "
    "is cancelled: the answer carries cancel_requested=false, and when a "
    "record exists the stored snapshot is returned with a message saying "
    "nothing was changed."
)

PAUSE_DESCRIPTION = (
    "Stop the download with this progress_id without cancelling it, or "
    "resume one that was stopped: the first call suspends the transfer, "
    "the next one continues it. The queue, the files already fetched and "
    "the active file's partial data are kept, and resuming continues that "
    "file from where it left off via an HTTP range request rather than "
    "starting over. Returns the snapshot, in the same shape as "
    "progress_get_status, plus 'pause_action': 'paused' or 'resumed' for "
    "what this call did. 'unavailable' when the id is unknown, the "
    "operation is not a download — a benchmark or a model import reports "
    "this rather than being paused — has already finished or is being "
    "cancelled. Downloads only."
)


def cancel_operation(progress_id: str) -> dict:
    """End one operation and have MSHCore undo what it did.

    Args:
        progress_id: Identifier returned by a tracked tool.

    Returns:
        dict: Final snapshot after the cleanup, or the stored record when the
        operation was not running.
    """
    job = registry.get(progress_id)

    if job is None:
        return _inert(progress_id, "cancel_requested", False)

    return workers.cancel(job)


def pause_operation(progress_id: str) -> dict:
    """Suspend one download, or continue a suspended one.

    Cancel and Stop are different operations. Cancel ends the task and has MSHCore
    undo it, and applies to both kinds. Stop only suspends a download and leaves the
    task intact, so it exists for downloads alone — they are the only operation MSHCore
    can pause and resume.

    Args:
        progress_id: Identifier returned by a tracked tool.

    Returns:
        dict: Snapshot afterwards, with ``pause_action``.
    """
    job = registry.get(progress_id)

    if job is None:
        return _inert(progress_id, "pause_action", "unavailable")

    return workers.pause(job)


def _register_panel_tools(apps: Apps) -> None:
    """Register the tools the panel itself calls over its bridge.

    These are the only read-and-control tools bound to the panel, and they are
    ``visibility=["app"]``: the view calls them, the model does not see them, and a
    call the view makes is answered back into its iframe rather than added to the
    conversation. That is what lets the panel poll twice a second without the
    conversation growing a progress bar per poll.

    They are named apart from the model's tools so the two surfaces cannot be
    confused, and each is a one-line delegation to the same function the model's
    equivalent calls.

    Args:
        apps: Extension the tools are added to.
    """

    @apps.tool(
        resource_uri=PROGRESS_URI,
        visibility=["app"],
        name="progress_panel_status",
        title="Read progress for the panel",
        description=(
            "Internal: the progress panel's own poll for the operation it is "
            "rendering. Returns the same snapshot as progress_get_status."
        ),
        annotations=ToolAnnotations(
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=False,
            open_world_hint=False,
        ),
    )
    @surface_core_errors
    def progress_panel_status(progress_id: str) -> dict:
        """Return one job's snapshot, for the panel.

        Args:
            progress_id: Identifier the panel adopted from its tool result.

        Returns:
            dict: The snapshot, or a not-found response.
        """
        return read_status(progress_id)

    @apps.tool(
        resource_uri=PROGRESS_URI,
        visibility=["app"],
        name="progress_panel_cancel",
        title="Cancel from the panel",
        description=f"Internal: the panel's Cancel button. {CANCEL_DESCRIPTION}",
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=True,
            idempotent_hint=True,
            open_world_hint=False,
        ),
    )
    @surface_core_errors
    def progress_panel_cancel(progress_id: str) -> dict:
        """Cancel one operation from the panel.

        Args:
            progress_id: Identifier the panel adopted from its tool result.

        Returns:
            dict: Final snapshot after the cleanup.
        """
        return cancel_operation(progress_id)

    @apps.tool(
        resource_uri=PROGRESS_URI,
        visibility=["app"],
        name="progress_panel_pause",
        title="Stop or resume from the panel",
        description=f"Internal: the panel's Stop button. {PAUSE_DESCRIPTION}",
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=False,
            open_world_hint=False,
        ),
    )
    @surface_core_errors
    def progress_panel_pause(progress_id: str) -> dict:
        """Suspend or continue one download from the panel.

        Args:
            progress_id: Identifier the panel adopted from its tool result.

        Returns:
            dict: Snapshot afterwards, with ``pause_action``.
        """
        return pause_operation(progress_id)


def _inert(progress_id: str, field: str, value: Any) -> dict:
    """Build the answer for a control aimed at an operation that is not running.

    The stored record is returned when there is one, so the caller learns how the
    operation ended rather than only that it is not running now.

    Args:
        progress_id: Identifier that was asked for.
        field: Outcome field the calling tool documents.
        value: Value marking the request as having had no effect.

    Returns:
        dict: Snapshot or not-found response, with the outcome field set.
    """
    record = load_snapshot(progress_id)

    if record is None:
        return {**_not_found(progress_id), field: value}

    return {
        **record,
        "found": True,
        field: value,
        "message": (
            f"{record.get('message') or 'Already finished'} — nothing was changed."
        ),
    }


def _started(job: Job, **extra: Any) -> dict:
    """Build a tracked tool's return value.

    ``progress_id`` comes first and is the only thing the panel needs; the rest
    tells the model what it has been given and what it still owes.

    Args:
        job: Job that was started.
        extra: Additional fields for the model.

    Returns:
        dict: Tool result carrying the identifier.
    """
    return {"progress_id": job.id, "status": job.snapshot()["status"], **extra}


BENCHMARK_CONTRACT = (
    "This benchmark has started and has NOT completed. This response is an "
    "acknowledgement, not the benchmark result, and it contains no measurements. "
    "Say that it has started, then end the turn and go to sleep — do not hold "
    "the turn open polling. When the user calls again, poll progress_get_status "
    "with this progress_id until its status is 'completed', 'failed' or "
    "'cancelled'. On 'completed', call benchmark_get_result with the same "
    "progress_id to read the measurements the run stored on disk, and use those "
    "to answer the user. Do not report the benchmark as done, and do not "
    "describe its results, before retrieving them."
)


def _benchmark_started(job: Job, **extra: Any) -> dict:
    """Build a benchmark tool's return value.

    Args:
        job: Job that was started.
        extra: Additional fields for the model.

    Returns:
        dict: Handle plus the contract the model has to follow.
    """
    return {
        **_started(job, **extra),
        "result_available": False,
        "next_step": (
            f"End the turn; when the user calls again, poll "
            f"progress_get_status(progress_id='{job.id}') until it reports a "
            f"terminal status, then call "
            f"benchmark_get_result(progress_id='{job.id}')."
        ),
        "contract": BENCHMARK_CONTRACT,
    }


ADD_MODEL_CONTRACT = (
    "This import has started and has NOT completed. This response is an "
    "acknowledgement, not the result. Say that it has started, then end the "
    "turn and go to sleep — do not hold the turn open polling. When the user "
    "calls again, poll progress_get_status with this progress_id until its "
    "status is 'completed', 'failed' or 'cancelled'. A completed import has "
    "nothing further to fetch: the model is registered and usable at once — "
    "confirm it with ollama_list_models if the user asks."
)


def _model_started(job: Job, **extra: Any) -> dict:
    """Build a model-import tool's return value.

    Args:
        job: Job that was started.
        extra: Additional fields for the model.

    Returns:
        dict: Handle plus the contract the model has to follow.
    """
    return {
        **_started(job, **extra),
        "next_step": (
            f"End the turn; when the user calls again, poll "
            f"progress_get_status(progress_id='{job.id}') until it reports a "
            f"terminal status. A completed import needs no further call."
        ),
        "contract": ADD_MODEL_CONTRACT,
    }


def _download_destination(manager: DownloadManager, directory: str) -> str | None:
    """Report the path the single queued file is being written to.

    MSHCore may rename a file to avoid overwriting one already on disk, so the
    name is read back from the queue rather than assumed from the URL.

    Args:
        manager: Manager holding the one-file queue.
        directory: Directory the session writes into.

    Returns:
        str | None: Full path to the file, or None when the queue is empty.
    """
    downloads = manager.get_status()["downloads"]

    if not downloads:
        return None

    return str(Path(directory) / downloads[0]["filename"])


def _register_download(
    apps: Apps,
    get_session: Callable[[str], Any],
    register_session: Callable[[str, Any], None] | None,
    release_session: Callable[[str], Any] | None,
) -> None:
    """Register the download tools bound to the panel.

    Both are starting tools: ``download_file`` builds its own one-file session
    and starts it, and ``download_start`` sets a session the plain queue tools
    built running. The session registry itself stays with the layer that owns
    it — ``main.py`` — handed in here as the ``register_session`` and
    ``release_session`` callbacks.

    Args:
        apps: Extension the tools are added to.
        get_session: Resolver for a download session id.
        register_session: Adds a new session to the owning layer's registry.
        release_session: Drops a session from the owning layer's registry.
    """

    @apps.tool(
        resource_uri=PROGRESS_URI,
        name="download_file",
        title="Download one file",
        description=(
            "Download a single URL. PRIMARY download tool and the only one "
            "needed for one file: it creates the session, queues the URL and "
            "starts the transfer with a live progress bar in one call, so "
            "never combine it with download_create_session, download_add or "
            "download_start. url must be http or https on a host from "
            "download_list_allowed_domains. destination_directory defaults to "
            "%LOCALAPPDATA%\\MSH\\downloads and is created with its parents if "
            "missing; a relative path resolves against this server's working "
            "directory. filename overrides the name taken from the URL, with "
            "any directory part of it stripped; max_retries is the total "
            "attempts per file, 3 by default. Returns immediately with a "
            "ticket carrying 'progress_id' (pass it only to "
            "progress_get_status, progress_pause or progress_cancel), "
            "'session_id' (generated, shaped 'auto-<8 hex>'; pass it only to "
            "the download_* queue tools if the transfer needs pausing, "
            "skipping or cancelling through them), 'destination' (the full "
            "path the file is being written to, under the name actually "
            "reserved), 'status' ('starting' — the first poll reports "
            "'running') and 'next_step' (a ready-made hint naming both "
            "identifiers). Those two identifiers are not interchangeable. "
            "Never overwrites: when the destination name is already taken the "
            "file is saved under a numbered variant, and 'destination' "
            "reports the name actually used. A rejected domain or an unusable "
            "directory fails before anything is queued or written."
        ),
        annotations=LONG_RUNNING,
    )
    @surface_core_errors
    def download_file(
        url: str,
        destination_directory: str = str(DOWNLOADS_DIRECTORY),
        filename: str | None = None,
        max_retries: int = 3,
    ) -> dict:
        """Download one URL, creating and starting its session in one call.

        Args:
            url: HTTP or HTTPS URL on an allowed domain.
            destination_directory: Directory the file is written into, created
                if needed. Defaults to %LOCALAPPDATA%\MSH\downloads.
            filename: Optional destination filename; taken from the URL when
                omitted, and numbered if that name is already taken.
            max_retries: Retry attempts before the transfer is marked failed.

        Returns:
            dict: A ticket carrying ``progress_id``, ``session_id``,
            ``destination`` and ``status``.

        Raises:
            ToolError: If the session could not be created or the transfer
                could not be started.
        """
        # The session is this tool's own bookkeeping, not something the caller
        # named, so the id is generated. It is still returned, because the
        # download_* tools act on a session and a caller may want to pause or
        # cancel this one.
        session_id = f"auto-{uuid4().hex[:8]}"

        manager = DownloadManager(
            download_directory=destination_directory,
            max_retries=max_retries,
        )
        register_session(session_id, manager)

        try:
            manager.add(url=url, filename=filename)
            job = workers.start_download(manager, session_id)
        except Exception:
            # A rejected domain or an unusable directory must not leave a
            # session behind holding a queue nothing will ever run.
            release_session(session_id)
            raise

        destination = _download_destination(manager, destination_directory)

        return _started(
            job,
            session_id=session_id,
            destination=destination,
            next_step=(
                f"Poll progress_get_status(progress_id='{job.id}') for "
                f"progress, or download_get_status(session_id="
                f"'{session_id}') for the queue's own view."
            ),
        )

    @apps.tool(
        resource_uri=PROGRESS_URI,
        name="download_start",
        title="Start downloading",
        description=(
            "STEP 3 of 3: begin transferring a session's queue and show a live "
            "progress bar in the conversation: per-file bars with transferred and "
            "total bytes and an overall percentage. Requires download_create_session "
            "and at least one download_add or download_add_many first, and fails "
            "when the queue is empty; session_id is that session's name, never a "
            "progress_id. Returns immediately with 'progress_id', 'status' "
            "('starting'), 'session_id' and 'note', and the transfer "
            "continues in the background — starting it is the whole of this tool's "
            "job, so there is no result to collect afterwards. Carry on with other "
            "work and call progress_get_status with the id whenever the outcome "
            "matters, or to confirm the files arrived. Fails when the "
            "session is already downloading, naming the progress_id to poll "
            "in the error. Restarting a session retries files that "
            "failed and leaves completed, skipped and cancelled ones alone."
        ),
        annotations=LONG_RUNNING,
    )
    @surface_core_errors
    def download_start(session_id: str) -> dict:
        """Start a session's queue with a progress bar.

        Args:
            session_id: Session created by download_create_session.

        Returns:
            dict: The ``progress_id`` and the session's status just after starting.

        Raises:
            ToolError: If this session is already downloading.
        """
        manager = get_session(session_id)
        running = registry.find_download(session_id)

        if running is not None:
            # Two bars over one queue would each offer a Cancel button for the same
            # work, and the first pressed would delete the files the other was
            # still reporting.
            raise ToolError(
                f"Download session '{session_id}' is already downloading as "
                f"{running.id}. Poll that progress bar, or cancel it before "
                f"starting again."
            )

        job = workers.start_download(manager, session_id)

        return _started(
            job,
            session_id=session_id,
            note=(
                "The download is running in the background. No result needs to be "
                "collected; poll progress_get_status with this progress_id when the "
                "outcome matters."
            ),
        )


def _register_benchmarks(apps: Apps) -> None:
    """Register the four benchmark tools bound to the panel.

    One tool per Benchmark Mode of MSHCore — ``single``, ``configs``,
    ``models`` and ``tournament`` — each starting exactly its own mode's
    benchmark on its own progress bar. There is deliberately no generic
    matrix tool: the mode a run belongs to is part of what the Agent is
    asking for, so it names the tool rather than a parameter.

    Args:
        apps: Extension the tools are added to.
    """

    @apps.tool(
        resource_uri=PROGRESS_URI,
        name="benchmark_ollama_single",
        title="Measure one model's speed",
        description=(
            "Benchmark Mode 'single' — the plain speed test. Benchmarks "
            "exactly ONE model under Ollama's own defaults over a shared "
            "prompt list, running it in the background with a live progress "
            "bar showing one row per prompt. Use this to answer 'how fast is "
            "this model?' — nothing is compared and no parameters are varied. "
            "For several parameter sets on one model use "
            "benchmark_ollama_configs; for several models use "
            "benchmark_ollama_models or benchmark_ollama_tournament. "
            "model is an installed Ollama model name or tag exactly as "
            "ollama_list_models prints it; prompts is the non-empty list of "
            "prompt strings the model answers — every measurement is taken "
            "against these, so prefer the smallest list that can tell the "
            "answer apart. include_output defaults to false and keeps only "
            "measurements — pass true to keep each prompt's generated text "
            "in the result. repetitions runs every prompt more than once and "
            "reports the run-to-run spread. Returns immediately with a dict "
            "of 'progress_id', 'status' ('starting'), 'result_available' "
            "(false), 'mode', 'models', 'configurations', 'next_step' and "
            "'contract' — an acknowledgement that the benchmark has started, "
            "NOT its results, which do not exist yet. Say that it has "
            "started, end the turn and go to sleep; when the user calls "
            "again, poll progress_get_status with that id until the status "
            "is 'completed', 'failed' or 'cancelled', then call "
            "benchmark_get_result with the same id to obtain the "
            "measurements. The model is unloaded afterwards whatever the "
            "outcome. Verify the model with ollama_list_models first."
        ),
        annotations=LONG_RUNNING,
    )
    @surface_core_errors
    def benchmark_ollama_single(
        model: str,
        prompts: list[str],
        include_output: bool = False,
        repetitions: int = 1,
    ) -> dict:
        """Start the single-model speed-test benchmark.

        Args:
            model: Installed Ollama model name or tag.
            prompts: Prompt strings the model answers.
            include_output: Whether to keep generated text alongside metrics.
            repetitions: How many times every prompt runs, from 1.

        Returns:
            dict: The ``progress_id`` to poll, and the contract to follow.

        Raises:
            ToolError: If the model name is blank or prompts is not a
                non-empty list of strings.
        """
        if not isinstance(model, str) or not model.strip():
            raise ToolError("A model name is required.")

        _validate_prompts(prompts)

        job = workers.start_benchmark_single(
            model=model.strip(),
            prompts=prompts,
            include_output=include_output,
            repetitions=repetitions,
        )

        return _benchmark_started(
            job,
            mode="single",
            models=[model.strip()],
            configurations=1,
        )

    @apps.tool(
        resource_uri=PROGRESS_URI,
        name="benchmark_ollama_configs",
        title="Compare configurations of one model",
        description=(
            "Benchmark Mode 'configs' — the configuration test. Benchmarks "
            "ONE model under two or more temporary parameter sets, all "
            "answering the same prompt list, with "
            "a live progress bar showing one weighted row per "
            "configuration. Use this to tune generation parameters — "
            "temperature, num_ctx, top_k, … — and learn which set is faster "
            "and whether the gap beats the noise; the configurations are "
            "applied per request and create nothing on disk (only "
            "ollama_configure_model bakes a variant). For one model's raw "
            "speed use benchmark_ollama_single; for comparing models use "
            "benchmark_ollama_models or benchmark_ollama_tournament. model "
            "is an installed Ollama model name or tag exactly as "
            "ollama_list_models prints it. configurations is the list to "
            "race: each entry a dict with optional 'name' (unnamed entries "
            "are listed under 'configuration_<position>') and 'options' (an "
            "Ollama options dict; omitted options mean Ollama's defaults). "
            "prompts is the shared prompt list every configuration answers "
            "— one list for the whole comparison is what makes the numbers "
            "comparable. include_output defaults to false; repetitions runs "
            "every prompt more than once per configuration and turns the "
            "significance verdict from null into real-or-noise. Returns "
            "immediately with 'progress_id', 'status' ('starting'), "
            "'result_available' (false), 'mode', 'models', "
            "'configurations' (how many will run), 'next_step' and "
            "'contract' — an acknowledgement, NOT the results. Say that it "
            "has started, end the turn and go to sleep; when the user calls "
            "again, poll progress_get_status with that id until terminal, "
            "then call benchmark_get_result with the same id for the "
            "side-by-side measurements and the significance assessment. "
            "Verify the model with ollama_list_models first."
        ),
        annotations=LONG_RUNNING,
    )
    @surface_core_errors
    def benchmark_ollama_configs(
        model: str,
        configurations: list[dict],
        prompts: list[str],
        include_output: bool = False,
        repetitions: int = 1,
    ) -> dict:
        """Start the one-model configuration comparison.

        Args:
            model: Installed Ollama model name or tag.
            configurations: Dicts with optional 'name' and 'options'.
            prompts: Prompt strings every configuration answers.
            include_output: Whether to keep generated text alongside metrics.
            repetitions: How many times every prompt runs per configuration.

        Returns:
            dict: The ``progress_id`` to poll, and the contract to follow.

        Raises:
            ToolError: If the arguments are malformed — checked here because
                the worker runs after this returns.
        """
        if not isinstance(model, str) or not model.strip():
            raise ToolError("A model name is required.")

        _validate_configurations(model.strip(), configurations)
        _validate_prompts(prompts)

        job = workers.start_benchmark_configs(
            model=model.strip(),
            configurations=configurations,
            prompts=prompts,
            include_output=include_output,
            repetitions=repetitions,
        )

        return _benchmark_started(
            job,
            mode="configs",
            models=[model.strip()],
            configurations=len(configurations),
        )

    @apps.tool(
        resource_uri=PROGRESS_URI,
        name="benchmark_ollama_models",
        title="Compare several models",
        description=(
            "Benchmark Mode 'models' — the model comparison. Benchmarks TWO "
            "or more models over the same prompt list under one shared "
            "configuration, with a live progress bar "
            "showing one weighted row per model. Use this to answer 'which "
            "model is fastest?' on like terms. To vary parameters per model, "
            "use benchmark_ollama_tournament instead; for one model alone, "
            "benchmark_ollama_single or benchmark_ollama_configs. models is "
            "the list of installed Ollama model names or tags to compare, "
            "each exactly as ollama_list_models prints it, in run order. "
            "config is the generation-parameter dict every model runs under "
            "— omit it (or pass {}) so every model races Ollama's defaults, "
            "which is the fairest cross-model question. prompts is the "
            "shared prompt list every model answers. include_output defaults "
            "to false; repetitions averages every prompt over several runs "
            "and reports the run-to-run spread. Models run one after "
            "another — each is unloaded before the next loads, so timings "
            "never compete for VRAM. Returns immediately with "
            "'progress_id', 'status' ('starting'), 'result_available' "
            "(false), 'mode', 'models', 'configurations', 'next_step' and "
            "'contract' — an acknowledgement, NOT the results. Say that it "
            "has started, end the turn and go to sleep; when the user calls "
            "again, poll progress_get_status with that id until terminal, "
            "then call benchmark_get_result with the same id for the "
            "side-by-side measurements and the 'across_models' significance "
            "verdict. Verify every model with ollama_list_models first."
        ),
        annotations=LONG_RUNNING,
    )
    @surface_core_errors
    def benchmark_ollama_models(
        models: list[str],
        prompts: list[str],
        config: dict | None = None,
        include_output: bool = False,
        repetitions: int = 1,
    ) -> dict:
        """Start the multi-model comparison under one shared configuration.

        Args:
            models: Installed model names or tags, in run order.
            prompts: Prompt strings every model answers.
            config: Generation parameters shared by every model. Optional.
            include_output: Whether to keep generated text alongside metrics.
            repetitions: How many times every prompt runs per model.

        Returns:
            dict: The ``progress_id`` to poll, and the contract to follow.

        Raises:
            ToolError: If fewer than two usable models were given, or the
                other arguments are malformed.
        """
        cleaned = _validate_models(models)
        _validate_options(config, "config")
        _validate_prompts(prompts)

        job = workers.start_benchmark_models(
            models=cleaned,
            prompts=prompts,
            config=config,
            include_output=include_output,
            repetitions=repetitions,
        )

        return _benchmark_started(
            job,
            mode="models",
            models=cleaned,
            configurations=len(cleaned),
        )

    @apps.tool(
        resource_uri=PROGRESS_URI,
        name="benchmark_ollama_tournament",
        title="Run a model tournament",
        description=(
            "Benchmark Mode 'tournament' — each model races the "
            "configuration paired with it. Use this when each "
            "model should be measured under its own best-fit settings — a "
            "different question from benchmark_ollama_models, where one "
            "shared configuration isolates the models themselves. models is "
            "the list of installed Ollama model names or tags racing, each "
            "exactly as ollama_list_models prints it, in run order. "
            "model_configs pairs them: a dict keyed by model name whose "
            "values are that model's Ollama options dict. A model missing "
            "from the pairing falls back to the shared config argument. "
            "prompts is the "
            "shared prompt list every model answers even though their "
            "settings differ. include_output defaults to false; repetitions "
            "averages every prompt over several runs. Models run one after "
            "another, each unloaded before the next loads. Returns "
            "immediately with 'progress_id', 'status' ('starting'), "
            "'result_available' (false), 'mode', 'models', "
            "'configurations', 'next_step' and 'contract' — an "
            "acknowledgement, NOT the results. Say that it has started, end "
            "the turn and go to sleep; when the user calls again, poll "
            "progress_get_status with that id until terminal, then call "
            "benchmark_get_result with the same id for the measurements and "
            "both significance halves ('by_model' and 'across_models'). "
            "Verify every model with ollama_list_models first."
        ),
        annotations=LONG_RUNNING,
    )
    @surface_core_errors
    def benchmark_ollama_tournament(
        models: list[str],
        prompts: list[str],
        model_configs: dict | None = None,
        config: dict | None = None,
        include_output: bool = False,
        repetitions: int = 1,
    ) -> dict:
        """Start the tournament: each model under its paired configuration.

        Args:
            models: Installed model names or tags, in run order.
            prompts: Prompt strings every model answers.
            model_configs: Per-model options keyed by model name. Optional.
            config: Shared fallback options for unpaired models. Optional.
            include_output: Whether to keep generated text alongside metrics.
            repetitions: How many times every prompt runs per model.

        Returns:
            dict: The ``progress_id`` to poll, and the contract to follow.

        Raises:
            ToolError: If fewer than two usable models were given, or the
                other arguments are malformed.
        """
        cleaned = _validate_models(models)

        if model_configs is None:
            model_configs = {}

        if not isinstance(model_configs, dict):
            raise ToolError("model_configs must be an object keyed by model name.")

        for name, entry in model_configs.items():
            if name not in cleaned:
                raise ToolError(
                    f"model_configs names '{name}', which is not among the "
                    f"models racing."
                )

            _validate_options(entry, f"model_configs['{name}']")

        _validate_options(config, "config")
        _validate_prompts(prompts)

        job = workers.start_benchmark_tournament(
            models=cleaned,
            prompts=prompts,
            model_configs=model_configs,
            config=config,
            include_output=include_output,
            repetitions=repetitions,
        )

        return _benchmark_started(
            job,
            mode="tournament",
            models=cleaned,
            configurations=len(cleaned),
        )


def _register_models(apps: Apps) -> None:
    """Register the model-import tool bound to the panel.

    ``ollama_add_model`` is the tracked form of MSHCore's model import: it
    starts a job, shows the bar, and returns the handle. Only the starting
    form lives here; there is no separate plain tool to migrate a session to.

    Args:
        apps: Extension the tool is added to.
    """

    @apps.tool(
        resource_uri=PROGRESS_URI,
        name="ollama_add_model",
        title="Import a local model file",
        description=(
            "Register a model weights file already on disk — typically a "
            ".gguf — with Ollama under a new name, making it usable by every "
            "other ollama_* tool. Primary tool for adopting a manually "
            "obtained model. model_path is a local filesystem path to that "
            "weights file, not a URL: download it first with download_file "
            "and pass the 'destination' that reported. Requires the Ollama "
            "service to be running. model_name is the new name to register; "
            "Ollama overwrites an existing model of that name without "
            "warning, so check ollama_list_models first. Ollama copies the "
            "weights into its own store, so this consumes disk space roughly "
            "equal to the file's size and the original file is left where it "
            "is. Runs in the background and shows a minimal live view in "
            "the conversation: the model's name and a status badge, and "
            "nothing else while it runs — no percentage, no status text, no "
            "Cancel button. Only a failure puts a line under the title, "
            "saying why the import failed. The read that opens the import "
            "is long and unmeasurable, so a bar would sit at zero and then "
            "race to 100, and the import is left to finish on its own. "
            "Returns immediately with 'progress_id', "
            "'status' ('starting'), 'model', 'next_step' and 'contract' — "
            "an acknowledgement that the import has started, NOT that it "
            "finished. Say that it has started, end the turn and go to "
            "sleep; when the user calls again, poll progress_get_status with "
            "that id until the status is 'completed', 'failed' or "
            "'cancelled'. A completed import has nothing further to fetch — "
            "the model is registered and usable at once. Fails immediately "
            "when the path is not a file."
        ),
        annotations=LONG_RUNNING,
    )
    @surface_core_errors
    def ollama_add_model(model_name: str, model_path: str) -> dict:
        """Import a local model file into Ollama with a progress bar.

        Args:
            model_name: Name to register the model under.
            model_path: Path to the model file on disk.

        Returns:
            dict: The ``progress_id`` to poll, and the contract to follow.

        Raises:
            ToolError: If the path is not an existing file — checked here
                because the worker runs after this returns, so a bad path
                would otherwise surface only on the progress bar.
        """
        if not Path(model_path).expanduser().is_file():
            raise ToolError(f"Model file not found: {model_path}")

        job = workers.start_add_model(
            model_name=model_name,
            model_path=model_path,
        )

        return _model_started(job, model=model_name)


def _validate_prompts(prompts: list[str]) -> None:
    """Check a benchmark's prompt list before the worker starts.

    MSHCore validates these itself, but on the worker thread — after the tool
    has returned and the progress bar exists. Doing it here means a malformed
    argument is a tool error the model can act on rather than a failed bar.

    Args:
        prompts: The prompts every configuration answers.

    Raises:
        ToolError: If the list is empty or holds a non-string.
    """
    if not isinstance(prompts, list) or not prompts:
        raise ToolError(
            "prompts is required: every configuration answers the same "
            "prompt list."
        )

    if not all(isinstance(prompt, str) for prompt in prompts):
        raise ToolError("prompts must be a list of strings.")


def _validate_options(options: dict | None, label: str) -> None:
    """Check one generation-options dictionary.

    Args:
        options: The options to check; None means "not given".
        label: Name of the argument, used in the error text.

    Raises:
        ToolError: If the value is neither None nor a dict.
    """
    if options is not None and not isinstance(options, dict):
        raise ToolError(f"{label} must be an object of Ollama options.")


def _validate_models(models: list[str]) -> list[str]:
    """Check a multi-model run's model list and return it trimmed.

    Args:
        models: Model names as given to the tool.

    Returns:
        list[str]: The names, stripped, in run order.

    Raises:
        ToolError: If fewer than two usable names were given.
    """
    if not isinstance(models, list) or not models:
        raise ToolError("A list of models is required.")

    cleaned = []

    for position, name in enumerate(models, start=1):
        if not isinstance(name, str) or not name.strip():
            raise ToolError(f"Model {position} must be a non-empty string.")

        cleaned.append(name.strip())

    if len(cleaned) < 2:
        raise ToolError(
            "At least two models are required to compare them; use "
            "benchmark_ollama_single or benchmark_ollama_configs for one."
        )

    return cleaned


def _validate_configurations(model: str, configurations: list[dict]) -> None:
    """Check a configuration comparison's configuration list.

    Prompts stay shared by design — one list for the whole comparison is what
    keeps the configurations' numbers comparable — so a configuration
    carrying a 'prompts' key of its own is rejected outright.

    Args:
        model: The model the configurations run under, for error text.
        configurations: The configurations as given to the tool.

    Raises:
        ToolError: If the list is empty, an entry is not usable, or one
            carries its own prompts.
    """
    if not isinstance(configurations, list) or not configurations:
        raise ToolError(
            f"At least one configuration is required to compare them under "
            f"'{model}'."
        )

    for index, configuration in enumerate(configurations, start=1):
        if not isinstance(configuration, dict):
            raise ToolError(
                f"Configuration {index} of '{model}' must be an object."
            )

        options = configuration.get("options", {})

        if not isinstance(options, dict):
            raise ToolError(
                f"Configuration {index} of '{model}' options must be an "
                f"object."
            )

        if "prompts" in configuration:
            raise ToolError(
                f"Configuration {index} of '{model}' carries 'prompts', "
                f"but prompts are shared across every configuration: pass "
                f"them as prompts instead."
            )
