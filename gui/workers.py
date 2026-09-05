"""Workers that run the tracked operations and keep their jobs current.

Every tracked tool follows the same shape. It creates a job, hands the work to a
background thread, and returns the job's identifier straight away — so the panel
has its id within milliseconds and the MCP request is never held open for the
length of the operation. That is what lets ``progress_get_status`` be answered
while a benchmark is still generating.

Each worker owns its job's whole lifecycle and reaches exactly one terminal
status:

    try:      run the MSHCore function, classify what it returned  → completed/failed
    except:   OperationCancelled → cancelled, anything else     → failed
    finally:  the job is finished either way

Where the progress figures come from is each worker's business and stops here.
Downloads read ``DownloadManager.get_status``, which is a real live API. Benchmarks
are handed their figures by MSHCore itself, through the ``on_progress`` callback it
invokes before every prompt and after every finished repetition; the worker only
reshapes those dicts into job steps.
"""

from __future__ import annotations

from collections.abc import Callable
import threading
import time
from typing import Any

# MSHCore is installed with pip and imported as a top-level package, the same
# way the server's other modules import it. Mixing spellings — say, `MSHCore.x`
# here but `Core.MSHCore.x` in main.py — would put two distinct copies of every
# class in memory, so a token handed to MSHCore and the OperationCancelled caught
# back from it would come from different copies, and `except
# OperationCancelled` would silently miss the cancellation.
from MSHCore.benchmark import ollama_runner
from MSHCore.cancellation import CancellationToken, OperationCancelled
from MSHCore.download_manager.manager import DownloadManager

from .jobs import (
    CANCELLED,
    COMPLETED,
    FAILED,
    RUNNING,
    SKIPPED,
    WAITING,
    Job,
    Metric,
    registry,
)

# How often a worker reads its progress source.
POLL_SECONDS = 0.4

# How long a cancellation waits for MSHCore to finish cleaning up.
CANCEL_TIMEOUT = 60.0

# How long a download worker waits for the manager's thread to exit after a
# cancellation, so the final snapshot describes the state after cleanup.
WORKER_STOP_TIMEOUT = 30.0

# DownloadManager's per-item statuses, mapped onto step states.
DOWNLOAD_STEP_STATES = {
    "waiting": WAITING,
    "connecting": RUNNING,
    "downloading": RUNNING,
    "retrying": RUNNING,
    "paused": RUNNING,
    "completed": COMPLETED,
    "failed": FAILED,
    "skipped": SKIPPED,
    "cancelled": CANCELLED,
}


def _spawn(job: Job, target: Callable[[], None], name: str) -> None:
    """Run a job's worker on a daemon thread.

    A thread that cannot be started — the process is out of them — would leave the
    job at ``starting`` with nothing to advance it, so that failure is the job's
    failure.

    Args:
        job: Job the worker belongs to.
        target: Zero-argument callable that runs the operation to a terminal state.
        name: Thread name, for debugging.
    """
    try:
        threading.Thread(target=target, name=name, daemon=True).start()
    except Exception as error:
        job.finish(
            FAILED,
            message="The operation could not be started.",
            error=str(error),
        )


def format_bytes(count: float | None) -> str:
    """Format a byte count for the panel.

    Args:
        count: Number of bytes, or None when the size is unknown.

    Returns:
        str: Size with a binary unit, or ``"?"`` when unknown.
    """
    if count is None:
        return "?"

    size = float(count)

    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.{0 if unit == 'B' else 1}f} {unit}"
        size /= 1024

    return f"{size:.1f} TB"


def format_speed(bytes_per_second: float | None) -> str | None:
    """Format a transfer rate for the panel.

    Args:
        bytes_per_second: Rate MSHCore measured, or None when it has none yet.

    Returns:
        str | None: Rate per second, or None when there is nothing to show —
        which is how a queue that has not started, or one that is paused,
        reports itself.
    """
    if not bytes_per_second:
        return None

    return f"{format_bytes(bytes_per_second)}/s"


# ============================================================
# Downloads
# ============================================================

def start_download(manager: DownloadManager, session_id: str) -> Job:
    """Begin a session's queue and follow it on a worker thread.

    Args:
        manager: Session manager to start.
        session_id: Session identifier, shown on the panel.

    Returns:
        Job: The job, already persisted, whose id the panel polls.
    """
    status = manager.get_status()
    filenames = [item["filename"] for item in status["downloads"]]

    job = Job(
        kind="download",
        title=f"Downloading {len(filenames)} file(s)",
        message=f"session {session_id}",
        session_id=session_id,
    )
    job.add_steps(filenames)
    # DownloadManager owns both controls: cancel() removes what the session
    # produced, pause()/resume() suspend the transfer and keep the queue.
    job.set_cancel(lambda reason: manager.cancel(reason=reason))
    job.set_pause(pause=manager.pause, resume=manager.resume)

    _spawn(
        job,
        lambda: _run_download(job, manager, session_id),
        name=f"download-{session_id}",
    )

    return job


def _run_download(job: Job, manager: DownloadManager, session_id: str) -> None:
    """Run one download to a terminal status.

    Args:
        job: Job to keep current.
        manager: Session to start and watch.
        session_id: Session identifier, for the session hook.
    """
    try:
        manager.start()
        job.begin()

        while True:
            status = manager.get_status()
            _apply_download_status(job, status)
            job.publish()

            if not status["running"]:
                break

            time.sleep(POLL_SECONDS)

        # A cancellation returns before the manager's thread has exited, and
        # MSHCore's cleanup runs on that thread. Waiting means the final snapshot
        # describes the state after the files were removed, not during.
        manager.wait_until_stopped(timeout=WORKER_STOP_TIMEOUT)

        final = manager.get_status()
        _apply_download_status(job, final)
        _finish_download(job, final)
    except Exception as error:
        # Including whatever manager.start() raises — an empty queue, a session
        # already cancelled. The job must reach a terminal status either way, or
        # the panel polls a running job for ever.
        job.finish(FAILED, message="The download failed.", error=str(error))
    finally:
        _release_session(session_id, manager)


def _finish_download(job: Job, final: dict) -> None:
    """Classify a stopped download and finish its job.

    Args:
        job: Job to finish.
        final: Manager status after the queue stopped and cleanup ran.
    """
    if final["cancelled"]:
        deleted = final.get("files_deleted", True)

        if deleted:
            # cancel() removes every file the session produced, including ones
            # that had finished, so a row still reading "completed" would point at
            # something no longer on disk.
            for index, item in enumerate(final["downloads"]):
                if item["status"] == "completed":
                    job.finish_step(index, state=CANCELLED, detail="removed")

        job.finish(
            CANCELLED,
            message=final.get("cancel_reason")
            or (
                "Cancelled; downloaded files removed."
                if deleted
                else "Cancelled; downloaded files kept."
            ),
        )
        return

    failed = [
        item["filename"]
        for item in final["downloads"]
        if item["status"] == "failed"
    ]

    if failed:
        job.finish(
            FAILED,
            message=f"{len(failed)} file(s) failed.",
            error=", ".join(failed),
        )
        return

    job.finish(COMPLETED, message="All files downloaded.")


def _apply_download_status(job: Job, status: dict) -> None:
    """Copy one manager reading onto a job.

    Args:
        job: Job to update.
        status: Return value of ``DownloadManager.get_status``.
    """
    downloaded_total = 0
    expected_total = 0
    sizes_known = True
    active_speed: float | None = None

    job.sync_paused(bool(status["paused"]))

    for index, item in enumerate(status["downloads"]):
        downloaded = item.get("downloaded") or 0
        total = item.get("total")
        state = item["status"]

        downloaded_total += downloaded

        if total:
            expected_total += total
        elif state not in ("completed", "skipped"):
            sizes_known = False

        percent = 100.0 if state == "completed" else (
            100.0 * downloaded / total if total else None
        )

        detail = (
            f"{format_bytes(downloaded)} / {format_bytes(total)}"
            if downloaded or total
            else state
        )

        # MSHCore measures the rate per item, so the row carries its own and the
        # chip under the bar shows whichever row is transferring.
        speed = format_speed(item.get("speed")) if state == "downloading" else None

        if speed is not None:
            detail = f"{detail} · {speed}"
            active_speed = item.get("speed")

        if state == "paused":
            detail = f"{format_bytes(downloaded)} · stopped"

        step_state = DOWNLOAD_STEP_STATES.get(state, WAITING)

        if step_state == RUNNING:
            # Includes paused: the transfer is suspended, not finished, and
            # closing the row would make the bar look like it moved on.
            job.start_step(index)
            job.update_step(index, percent=percent, detail=detail, error=item.get("error"))
        elif step_state == WAITING:
            job.update_step(index, detail=detail)
        else:
            job.finish_step(index, state=step_state, detail=detail, error=item.get("error"))

    if sizes_known and expected_total:
        job.set_percent(100.0 * downloaded_total / expected_total)

    metrics = [
        Metric("downloaded", format_bytes(downloaded_total)),
        Metric("total", format_bytes(expected_total) if expected_total else "?"),
        Metric(
            "file",
            f"{min(status['current_index'] + 1, status['total_files'])}"
            f"/{status['total_files']}",
        ),
    ]

    # Only while something is actually transferring: a rate left on the panel
    # after the queue stopped would read as though it still were.
    rate = format_speed(active_speed)

    if rate is not None:
        metrics.append(Metric("speed", rate))

    job.set_metrics(metrics)


# Set by the server layer so a cancelled session can be dropped from wherever it
# lives. Kept as a hook rather than an import so this module does not depend on
# main.py's registry.
release_session: Callable[[str], Any] | None = None


def _release_session(session_id: str, manager: DownloadManager) -> None:
    """Drop a closed session from the owning registry.

    A cancelled or closed session has nothing left to continue from, so its id is
    freed. One that finished normally stays: its status is still worth reading.

    Args:
        session_id: Session that stopped.
        manager: Manager to ask whether it closed.
    """
    if release_session is None:
        return

    try:
        closed = manager.get_status()["closed"]
    except Exception:  # pragma: no cover - defensive
        closed = True

    if closed:
        release_session(session_id)


def note_download_ended(session_id: str, reason: str) -> bool:
    """Tell a session's progress bar that the session was ended elsewhere.

    ``download_cancel`` and ``download_close_session`` act on the session, not on
    the bar. The worker would notice on its next reading anyway, but until then the
    panel keeps offering Cancel and Stop for a queue that has gone.

    Args:
        session_id: Session that was ended.
        reason: Why, shown on the panel.

    Returns:
        bool: True when a job was tracking that session.
    """
    job = registry.find_download(session_id)

    if job is None:
        return False

    job.request_cancel(reason)
    job.wait(CANCEL_TIMEOUT)

    return True


# ============================================================
# Benchmarks
# ============================================================

def start_benchmark(
    experiments: list[dict],
    shared_prompts: list[str] | None,
    include_output: bool,
    repetitions: int = 1,
) -> Job:
    """Begin a benchmark matrix and run it on a worker thread.

    Calls ``ollama_runner.run_benchmark``, so MSHCore normalises the matrix
    and this layer has one result shape to classify. One row is created per
    model-configuration pair before MSHCore runs — one row per prompt
    instead when the matrix is a single pair, so the panel shows the shape
    of the work rather than an empty indeterminate bar.

    Args:
        experiments: One dict per model, in run order: 'model' plus optional
            'configurations', each carrying 'name', 'options' and an optional
            'prompts' list only that configuration answers.
        shared_prompts: Prompts every configuration answers before its own,
            or None when every configuration carries its own.
        include_output: Whether to include generated text in the results.
        repetitions: How many times every prompt runs per configuration, from 1.

    Returns:
        Job: The job, already persisted, whose id the panel polls.
    """
    prompts = list(shared_prompts or [])
    pairs = _matrix_pairs(experiments)
    models = _matrix_models(experiments)
    single = len(pairs) == 1 and bool(prompts)

    job = Job(
        kind="benchmark",
        title=(
            f"Benchmarking {models[0]}"
            if single or len(models) == 1
            else f"Comparing {len(models)} model(s)"
        ),
        message=(
            f"{len(prompts)} prompt(s) · {pairs[0][1]}"
            if single
            else f"{len(pairs)} configuration(s) · {len(prompts)} prompt(s) each"
        ),
    )

    # The rows exist before MSHCore runs, so the very first poll shows the shape of
    # the work rather than an empty indeterminate bar.
    if single:
        job.add_steps([f"prompt {index}" for index in range(1, len(prompts) + 1)])
    else:
        multi_model = len(models) > 1
        job.add_steps(
            [_pair_label(model, name, multi_model) for model, name in pairs],
            weight=float(len(prompts) or 1),
        )

    token = CancellationToken()
    job.set_cancel(token.cancel)

    def run(cancellation: CancellationToken) -> dict:
        return ollama_runner.run_benchmark(
            experiments=experiments,
            shared_prompts=shared_prompts or None,
            include_output=include_output,
            cancellation=cancellation,
            repetitions=repetitions,
            on_progress=_on_progress(
                job,
                rows=None
                if single
                else {(model, name): index for index, (model, name) in enumerate(pairs)},
            ),
        )

    _spawn(
        job,
        lambda: _run_benchmark(
            job=job,
            token=token,
            single=single,
            run=run,
        ),
        name="benchmark",
    )

    return job


def _matrix_pairs(experiments: list[dict]) -> list[tuple[str, str]]:
    """List the model-configuration pairs a matrix will run, in run order.

    Mirrors MSHCore's own normalisation so the rows can be named before it
    runs: a model with no configurations runs once under 'default', and a
    configuration without a name takes its position.

    Args:
        experiments: The matrix as given to the worker.

    Returns:
        list[tuple[str, str]]: One (model, configuration name) per pair.
    """
    pairs: list[tuple[str, str]] = []

    for experiment in experiments:
        model = experiment["model"]
        configurations = experiment.get("configurations")

        if configurations is None:
            pairs.append((model, "default"))
            continue

        for index, configuration in enumerate(configurations, start=1):
            name = (
                configuration.get("name")
                if isinstance(configuration, dict)
                else None
            )
            pairs.append(
                (
                    model,
                    name if isinstance(name, str) and name else f"configuration_{index}",
                )
            )

    return pairs


def _matrix_models(experiments: list[dict]) -> list[str]:
    """List a matrix's distinct model names, in run order.

    Args:
        experiments: The matrix as given to the worker.

    Returns:
        list[str]: First occurrence of each model name.
    """
    models: list[str] = []

    for experiment in experiments:
        model = experiment["model"]

        if model not in models:
            models.append(model)

    return models


def _pair_label(model: str, name: str, multi_model: bool) -> str:
    """Name one model-configuration pair the way a panel row shows it.

    A matrix spanning several models can carry two configurations of the
    same name — two models' defaults, say — so there the model's name
    prefixes the configuration's, keeping every row distinct. The same rule
    the benchmark history applies to its stored labels.

    Args:
        model: The pair's model name.
        name: The pair's configuration name.
        multi_model: Whether the matrix spans more than one model.

    Returns:
        str: The label the row is shown under.
    """
    if multi_model and model != name:
        return f"{model} / {name}"

    return name


def _on_progress(
    job: Job,
    rows: dict[tuple[str, str], int] | None,
) -> Callable[[dict], None]:
    """Build the callback MSHCore invokes as a benchmark's steps advance.

    MSHCore calls it once before every prompt and once after every finished
    repetition, with a dict carrying which model and configuration (name and
    position of each), which prompt and repetition, how many of both, and how
    many steps the whole run has completed and will run in total. It carries no
    success information: a failed repetition is indistinguishable from a
    finished one here, so every row the callback closes is provisional and
    ``_close_benchmark_steps`` rewrites the states from the result afterwards.
    MSHCore already absorbs callback failures — an exception is logged and
    dropped rather than raised into the run — so this reshaping adds no
    guarding of its own.

    Args:
        job: Job whose rows the callback keeps current.
        rows: (model, configuration name) to row index, for a matrix. None
            when the rows are prompts of a single test, which the prompt
            index addresses directly.

    Returns:
        Callable[[dict], None]: The callback to hand to MSHCore.
    """
    def on_progress(step: dict) -> None:
        index = step.get("prompt_index")

        if not isinstance(index, int) or index < 1:
            return

        starting = step.get("phase") == "prompt_start"
        last_repetition = (
            step.get("repetition") == step.get("repetition_count")
        )

        if rows is None:
            if starting:
                job.start_step(index - 1, detail="running")
            elif last_repetition:
                job.finish_step(index - 1, state=COMPLETED)

            # Same persistence duty as the matrix path below: the callback is
            # the only thing moving this benchmark's rows.
            job.publish()

            return

        key = (step.get("model"), step.get("configuration"))
        row = rows.get(key)

        if row is None:
            return

        job.start_step(row)
        job.update_step(row, percent=_row_percent(step))

        if not starting and last_repetition:
            # The configuration's last repetition just finished. The row still
            # reads completed even when prompts failed — the result settles it.
            job.finish_step(row, state=COMPLETED)

        # The rows advance from this callback alone, so persisting here is what
        # keeps the record on disk current while the run continues — publish
        # itself throttles to twice a second.
        job.publish()

    return on_progress


def _row_percent(step: dict) -> float | None:
    """Derive a comparison row's percentage from one progress step.

    Args:
        step: Progress dict MSHCore emitted.

    Returns:
        float | None: Percent complete for the step's configuration, or None
        when the counts are missing or unusable.
    """
    prompt_index = step.get("prompt_index")
    prompt_count = step.get("prompt_count")
    repetition = step.get("repetition")
    repetition_count = step.get("repetition_count")

    if not all(
        isinstance(value, int) and value is not None
        for value in (prompt_index, prompt_count, repetition, repetition_count)
    ):
        return None

    if prompt_count <= 0 or repetition_count <= 0:
        return None

    done = (prompt_index - 1) * repetition_count + repetition

    return 100.0 * done / (prompt_count * repetition_count)


def _run_benchmark(
    job: Job,
    token: CancellationToken,
    single: bool,
    run: Callable[[CancellationToken], dict],
) -> None:
    """Run one benchmark to a terminal status.

    Args:
        job: Job to keep current.
        token: Cancellation token the panel's Cancel button sets.
        single: Whether the rows are prompts rather than tests.
        run: Runs the MSHCore benchmark with the given cancellation token and
            returns its result. Called on this worker thread.
    """
    try:
        job.begin()

        result = run(token)
    except OperationCancelled as error:
        job.finish(CANCELLED, message=str(error))
        return
    except Exception as error:
        job.finish(FAILED, message="The benchmark failed.", error=str(error))
        return

    _close_benchmark_steps(job, result=result, single=single)

    # MSHCore records a failed prompt as a result entry rather than raising, so a run
    # against a model that does not exist returns normally with every prompt
    # failed. Classifying the result is what turns that into a failure.
    error = _benchmark_error(result)

    if error is not None:
        job.finish(FAILED, message="Every prompt failed.", error=error)
        return

    # The measurements are the point of the benchmark, so they are stored with the
    # job rather than discarded. The model fetches them once the status is
    # completed; they are deliberately not repeated in every progress snapshot.
    # The comparison shape is what the benchmark history expects, so it is stored
    # whole and a single-test run is unwrapped only when it is read back.
    job.finish(
        COMPLETED,
        message="Finished. Retrieve the measurements with benchmark_get_result.",
        result=result,
    )


def _business_result(result: dict) -> dict:
    """Shape a stored comparison result as the benchmark's deliverable.

    A single test is one configuration or one model, so its result is unwrapped
    from the comparison wrapper to that test's own measurements. A comparison
    keeps MSHCore's own shape. Applied when the result is fetched rather than
    when it is stored: the benchmark history keeps every comparison whole.

    Args:
        result: Return value of ``ollama_runner.run_benchmark``.

    Returns:
        dict: The measurements the model reads.
    """
    tests = result.get("tests") or []

    if len(tests) == 1:
        model = result.get("model") or (result.get("models") or [None])[0]

        return {"model": model, **tests[0]}

    return result


def _benchmark_error(result: dict) -> str | None:
    """Report the error when a benchmark produced no successful prompt.

    A run where some prompts succeeded is a success: those measurements are real,
    and the failed rows carry their own errors.

    Args:
        result: Return value of ``ollama_runner.run_benchmark``.

    Returns:
        str | None: The first error found, or None when the run succeeded.
    """
    entries = [
        entry
        for test in result.get("tests") or []
        for entry in test.get("results") or []
    ]

    if not entries or any(entry.get("success") is not False for entry in entries):
        return None

    for entry in entries:
        if entry.get("error"):
            return str(entry["error"])

    return "No prompt produced a result."


def _close_benchmark_steps(job: Job, result: dict, single: bool) -> None:
    """Settle every row from the returned result.

    The rows advance from MSHCore's progress callback while the run is in flight,
    which is best-effort: the callback carries no success information, so a
    closed row always reads completed. The result is authoritative, so it closes
    every row again with its true state once the run is over.

    Args:
        job: Job whose rows are being closed.
        result: Return value of ``ollama_runner.run_benchmark``.
        single: Whether the rows are prompts rather than configurations.
    """
    tests = result.get("tests") or []

    if single:
        entries = tests[0].get("results") or [] if tests else []

        for index, entry in enumerate(entries):
            job.finish_step(
                index,
                state=FAILED if entry.get("success") is False else COMPLETED,
                error=entry.get("error"),
            )

        # More rows than results means the run stopped early without raising.
        for index in range(len(entries), job.step_count()):
            job.finish_step(index, state=SKIPPED, detail="not run")

        return

    # One row per configuration, in the order they were queued — which is the
    # order MSHCore runs and returns them in.
    for index in range(job.step_count()):
        if index >= len(tests):
            job.finish_step(index, state=SKIPPED, detail="not run")
            continue

        entries = tests[index].get("results") or []
        failed = sum(1 for entry in entries if entry.get("success") is False)

        job.finish_step(
            index,
            state=FAILED if entries and failed == len(entries) else COMPLETED,
            detail=f"{len(entries)}/{len(entries)}",
        )


# ============================================================
# Controls
# ============================================================

def cancel(job: Job) -> dict:
    """Cancel a job and wait for MSHCore to finish cleaning up.

    Args:
        job: Job to cancel.

    Returns:
        dict: Final snapshot, taken after the cleanup so it describes what
        happened rather than what was asked for. ``cleanup_complete`` is false only
        when the wait ran out with MSHCore still working.
    """
    requested = job.request_cancel("Cancelled from the progress panel")
    complete = job.wait(CANCEL_TIMEOUT) if requested else True

    snapshot = job.snapshot()
    snapshot["cancel_requested"] = requested
    snapshot["cleanup_complete"] = complete

    if not complete:
        snapshot["message"] = (
            "Cancelled; MSHCore is still cleaning up. The task will not resume."
        )

    return snapshot


def pause(job: Job) -> dict:
    """Suspend a download, or continue a suspended one.

    Args:
        job: Job to suspend or continue.

    Returns:
        dict: Snapshot afterwards, with ``pause_action`` set to ``paused``,
        ``resumed`` or ``unavailable``.
    """
    action = job.toggle_pause()

    snapshot = job.snapshot()
    snapshot["pause_action"] = action

    if action == "unavailable":
        snapshot["message"] = (
            "This operation cannot be stopped without cancelling it."
        )

    return snapshot
