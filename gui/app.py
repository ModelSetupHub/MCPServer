"""The progress panel: its MCP Apps extension, and the tools that read it.

MCP Apps (``io.modelcontextprotocol/ui``) is how a tool result carries a UI: the
tool advertises ``_meta.ui.resourceUri``, which points at a ``ui://`` HTML
resource the client renders inline in the conversation, in a sandboxed iframe.

That binding is what draws a progress bar, so it decides which tools may have it.
A tool bound to the panel gets a new panel every time the *model* calls it, since
each call is a new tool result in the conversation for the host to render. So only
the three tools that start an operation are bound to it — one call, one job, one
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
from typing import Any, TypeVar

from mcp.server.apps import Apps
from mcp.server.mcpserver.exceptions import MCPServerError, ToolError
from mcp.types import ToolAnnotations

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
            the owning layer can drop a cancelled session.

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
    _register_download(apps, get_session)
    _register_benchmarks(apps)

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
            "operation's recorded state, it does not wait for it. Returns status — "
            "'starting' or 'running' while the work continues, then one of "
            "'completed', 'failed' or 'cancelled' — with an overall percentage, a "
            "message, per-step rows, and an error when there is one. For a "
            "benchmark, 'result_available': true on a completed run means the "
            "measurements are ready to fetch with benchmark_get_result; the "
            "measurements are never included here. Do not poll in a held-open "
            "loop: after starting an operation, end the turn and go to sleep, "
            "then call this when the user next prompts, repeating only if the "
            "status is not yet terminal — the operation's own progress bar keeps "
            "updating itself either way. Answers while the operation runs and "
            "afterwards, including after this server restarts. An unknown id "
            "reports found=false rather than another operation's progress."
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
            "ollama_run_test_with_progress or ollama_run_benchmark_with_progress, "
            "by its progress_id. This is where a benchmark's actual results come "
            "from: the starting tool returns only a handle, and progress_get_status "
            "reports only progress. Call this once progress_get_status reports "
            "status='completed' and result_available=true — earlier it reports that "
            "the benchmark is still running, and for a failed or cancelled run it "
            "reports that there are no measurements. Returns the same per-prompt "
            "timings, token counts and averaged summary the synchronous "
            "ollama_run_test and ollama_run_benchmark return."
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
    # to the shape its synchronous tool returns, here rather than at write time.
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
    "The download or benchmark stops at its next safe point, MSHCore removes "
    "everything it created — partial and completed downloads, a loaded "
    "model — and records a cancelled entry in the execution log. For a "
    "download the session is removed too, so downloading the same files "
    "again means creating it fresh. Cannot be undone; to suspend a download "
    "and keep it, use progress_pause. Reports cancel_requested=false when "
    "the operation was not running."
)

PAUSE_DESCRIPTION = (
    "Stop the download with this progress_id without cancelling it, or "
    "resume one that was stopped. The queue, the files already fetched and "
    "the partial data are kept, and resuming continues the active file from "
    "where it left off via an HTTP range request. Downloads only: a "
    "benchmark reports pause_action='unavailable' rather than being "
    "cancelled."
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


def _register_download(apps: Apps, get_session: Callable[[str], Any]) -> None:
    """Register the download tool bound to the panel.

    Args:
        apps: Extension the tool is added to.
        get_session: Resolver for a download session id.
    """

    @apps.tool(
        resource_uri=PROGRESS_URI,
        name="download_start_with_progress",
        title="Start downloading with a progress bar",
        description=(
            "Start processing a session's queue and show a live progress bar in the "
            "conversation: per-file bars with transferred and total bytes and an "
            "overall percentage. Returns immediately with a progress_id, and the "
            "transfer continues in the background — starting it is the whole of this "
            "tool's job, so there is no result to collect afterwards. Carry on with "
            "other work and call progress_get_status with the id whenever the "
            "outcome matters, or to confirm the files arrived. Queue every file "
            "first, and prefer this over download_start whenever a human is "
            "watching."
        ),
        annotations=LONG_RUNNING,
    )
    @surface_core_errors
    def download_start_with_progress(session_id: str) -> dict:
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
    """Register the benchmarking tools bound to the panel.

    Args:
        apps: Extension the tools are added to.
    """

    @apps.tool(
        resource_uri=PROGRESS_URI,
        name="ollama_run_test_with_progress",
        title="Start benchmarking one configuration, with a progress bar",
        description=(
            "Start a benchmark of one model under one set of generation parameters, "
            "running it in the background and showing a live progress bar in the "
            "conversation with one row per prompt. Returns immediately with a "
            "progress_id — an acknowledgement that the benchmark has started, NOT "
            "its results, which do not exist yet. Say that it has started, end the "
            "turn and go to sleep; when the user calls again, poll "
            "progress_get_status with that id until the status is 'completed', "
            "'failed' or 'cancelled', then call benchmark_get_result with the same "
            "id to obtain the measurements. "
            "The benchmark is not finished, and its results cannot be reported, "
            "until that retrieval. repetitions runs every prompt more than once "
            "and reports the run-to-run spread with the means; each repetition is "
            "a full generation, so the run costs prompts x repetitions "
            "generations. Prefer this whenever a human is watching, since "
            "each prompt is a full generation; use the synchronous ollama_run_test "
            "when the measurements are wanted in one call and no progress bar is "
            "needed."
        ),
        annotations=LONG_RUNNING,
    )
    @surface_core_errors
    def ollama_run_test_with_progress(
        model_name: str,
        prompts: list[str],
        config: dict | None = None,
        name: str = "test",
        include_output: bool = False,
        repetitions: int = 1,
    ) -> dict:
        """Start a benchmark of one configuration.

        Args:
            model_name: Model name or tag to benchmark.
            prompts: Prompts to run, in order.
            config: Optional generation options, for example
                {"temperature": 0.7, "num_ctx": 4096}.
            name: Label recorded with the results.
            include_output: Whether to keep generated text alongside metrics.
            repetitions: How many times every prompt runs, from 1.

        Returns:
            dict: The ``progress_id`` to poll, and the contract to follow.
        """
        job = workers.start_benchmark(
            experiments=[
                {
                    "model": model_name,
                    "configurations": [
                        {"name": name, "options": dict(config or {})}
                    ],
                }
            ],
            shared_prompts=prompts,
            include_output=include_output,
            repetitions=repetitions,
        )

        return _benchmark_started(job, model=model_name, prompts=len(prompts))

    @apps.tool(
        resource_uri=PROGRESS_URI,
        name="ollama_run_benchmark_with_progress",
        title="Start a benchmark matrix, with a progress bar",
        description=(
            "Start a benchmark matrix of models, configurations and prompts, "
            "running it in the background and showing a live progress bar in the "
            "conversation with one row per model-configuration pair. experiments "
            "is a list of dicts, one per model in run order: 'model' (required) "
            "and 'configurations' (optional; each a dict with 'name', 'options' "
            "and an optional 'prompts' list only that configuration answers). "
            "Every configuration answers shared_prompts before its own; a model "
            "with no configurations runs once under its defaults. One model "
            "carrying several configurations compares parameter sets; several "
            "models under one shared configuration compare models. Models run "
            "one after another — each is unloaded before the next loads, so "
            "timings never compete for VRAM. Returns immediately with a "
            "progress_id — an acknowledgement that the benchmark has started, "
            "NOT its results, which do not exist yet. Say that it has started, "
            "end the turn and go to sleep; when the user calls again, poll "
            "progress_get_status with that id until the status is 'completed', "
            "'failed' or 'cancelled', then call benchmark_get_result with the "
            "same id to obtain the side-by-side measurements and the two-way "
            "'significance' assessment ('by_model' and 'across_models'). "
            "repetitions runs every prompt more than once per configuration and "
            "reports the run-to-run spread, which is also what turns the "
            "significance verdicts on. Total time is every pair's prompt count "
            "multiplied by repetitions, which is why this runs asynchronously; "
            "use the synchronous ollama_run_benchmark when the measurements are "
            "wanted in one call. Verify every model with ollama_list_models "
            "first, and prefer the smallest prompt list that can tell the "
            "subjects apart."
        ),
        annotations=LONG_RUNNING,
    )
    @surface_core_errors
    def ollama_run_benchmark_with_progress(
        experiments: list[dict],
        shared_prompts: list[str] | None = None,
        include_output: bool = False,
        repetitions: int = 1,
    ) -> dict:
        """Start a benchmark across a matrix of models and configurations.

        Args:
            experiments: One dict per model, shaped as
                {"model": "llama3", "configurations": [{"name": "warm",
                "options": {"temperature": 0.9}}]}.
            shared_prompts: Prompts every configuration answers.
            include_output: Whether to keep generated text alongside metrics.
            repetitions: How many times every prompt runs per configuration,
                from 1.

        Returns:
            dict: The ``progress_id`` to poll, and the contract to follow.

        Raises:
            ToolError: If no experiment was given, or one is malformed, or
                the matrix names no prompts at all. Checked here because the
                worker runs after this returns, so a bad argument would
                otherwise surface only on the progress bar.
        """
        experiments = _normalise_experiments(experiments, shared_prompts)

        job = workers.start_benchmark(
            experiments=experiments,
            shared_prompts=shared_prompts,
            include_output=include_output,
            repetitions=repetitions,
        )

        return _benchmark_started(
            job,
            models=sorted({experiment["model"] for experiment in experiments}),
            configurations=sum(
                len(experiment["configurations"])
                for experiment in experiments
            ),
        )


def _normalise_experiments(
    experiments: list[dict],
    shared_prompts: list[str] | None,
) -> list[dict]:
    """Validate the matrix before the worker starts.

    MSHCore normalises and validates these itself, but it does so on the worker
    thread — after the tool has returned. Doing it here means a malformed
    experiment is a tool error the model can act on, and a matrix that names
    no prompts fails now rather than as a failed progress bar.

    Args:
        experiments: Experiments as given to the tool.
        shared_prompts: Prompts every configuration answers, for checking
            that each configuration ends up with work to do.

    Returns:
        list[dict]: The experiments as given, verified.

    Raises:
        ToolError: If the list is empty or an entry is not usable.
    """
    if not isinstance(experiments, list) or not experiments:
        raise ToolError("At least one experiment is required.")

    if shared_prompts is not None and (
        not isinstance(shared_prompts, list)
        or not all(isinstance(prompt, str) for prompt in shared_prompts)
    ):
        raise ToolError("shared_prompts must be a list of strings.")

    for position, experiment in enumerate(experiments, start=1):
        if not isinstance(experiment, dict):
            raise ToolError(f"Experiment {position} must be an object.")

        model = experiment.get("model")

        if not isinstance(model, str) or not model.strip():
            raise ToolError(f"Experiment {position} must name a model.")

        configurations = experiment.get("configurations")

        if configurations is None:
            if not shared_prompts:
                raise ToolError(
                    f"Experiment '{model}' has nothing to run: give "
                    f"shared_prompts, or configurations with 'prompts' of "
                    f"their own."
                )

            continue

        if not isinstance(configurations, list) or not configurations:
            raise ToolError(
                f"Experiment '{model}' must carry a non-empty configurations "
                f"list, or none at all."
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

            prompts = configuration.get("prompts")

            if prompts is not None and (
                not isinstance(prompts, list)
                or not all(isinstance(prompt, str) for prompt in prompts)
            ):
                raise ToolError(
                    f"Configuration {index} of '{model}' prompts must be a "
                    f"list of strings."
                )

            if not prompts and not shared_prompts:
                raise ToolError(
                    f"Configuration {index} of '{model}' has nothing to run: "
                    f"give shared_prompts, or a 'prompts' list of its own."
                )

    return experiments
