"""ModelSetupHub MCP server.

Serves the ``MSHCore`` package (the ``Core`` submodule) over the Model Context
Protocol. Every tool is a thin pass-through to an existing MSHCore function; no
business logic lives here. The tool ``description`` strings and
``INSTRUCTIONS`` below are the model's only guidance, so they carry each
tool's prerequisites, identifier provenance, side effects and pagination
limits.

MSHCore must be installed with pip before the server can start (see
README.md): from the repository root run ``python utils/install_mshcore.py``
or ``pip install ./Core``.

Run over stdio:

    python main.py

Client configuration:

    {
      "mcpServers": {
        "modelsetuphub": {
          "command": "python",
          "args": ["C:/path/to/main.py"]
        }
      }
    }
"""

from collections.abc import Callable
import functools
from pathlib import Path
import threading
from typing import Any, TypeVar

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import MCPServerError, ToolError
from mcp.types import ToolAnnotations

PROJECT_ROOT = Path(__file__).resolve().parent

try:
    import MSHCore  # noqa: F401
except ImportError as error:
    raise RuntimeError(
        "The MSHCore package is not installed. Install it with pip before "
        f"starting this server: run 'python utils/install_mshcore.py' or "
        f"'pip install ./Core' in {PROJECT_ROOT}."
    ) from error

from MSHCore import logging as core_logging  # noqa: E402
from MSHCore.benchmark import history  # noqa: E402
from MSHCore.system import hardware, scanner  # noqa: E402

# The queue manager stays constructed here: the registry below holds its
# instances and download_create_session builds one per named session.
from MSHCore.download_manager.manager import DownloadManager  # noqa: E402

# Where a queue's downloads land unless a tool call names somewhere else. Read
# from MSHCore so this layer cannot disagree with the manager's own default:
# both are %LOCALAPPDATA%\MSH\downloads, a per-user directory needing no
# elevation.
from MSHCore.paths import DOWNLOADS_DIRECTORY  # noqa: E402

DEFAULT_DOWNLOAD_DIRECTORY = str(DOWNLOADS_DIRECTORY)

# The whitelist lives in its own module so the manager and the downloader
# validate against one list; it is read from there rather than through either of
# them. `allowed_sources` renders it for a client, wildcard subdomains included.
from MSHCore.download_manager.sources import allowed_sources  # noqa: E402
from MSHCore.ollama import model, runtime  # noqa: E402
from MSHCore.python import environment, installer, tools  # noqa: E402

# The in-chat progress panel, the tools that draw it, and the plain tools that
# read one afterwards. Every frontend file lives under gui/; this layer hands it
# the session registry's three operations, so the starting tools bound to the
# panel — download_file and download_start — share the sessions the plain
# download tools act on rather than keeping a registry of their own.
from gui import (  # noqa: E402
    create_progress_app,
    note_download_ended,
    register_progress_tools,
)

SERVER_NAME = "modelsetuphub"
SERVER_TITLE = "ModelSetupHub"
SERVER_VERSION = "0.1.0"

INSTRUCTIONS = """\
ModelSetupHub manages local AI environments on the machine this server runs
on: hardware discovery, the Ollama service and its models, model
benchmarking, Python interpreters, packages and scripts, and downloads
restricted to a fixed domain whitelist.

## Choosing a tool

- One hardware metric: system_get_gpu_info, system_get_storage_info,
  system_get_memory_info or system_get_cuda_version. Whole profile:
  system_scan, which on Windows makes eight subprocess calls and can take
  tens of seconds — do not call it for a single number. system_scan reports
  memory and storage in GiB; the narrow tools report raw bytes.
- Generate text: ollama_run_model. Measure speed: benchmark_run —
  one model under several configurations, several models under one shared
  configuration, or any mix, as the experiments matrix. Only
  ollama_configure_model creates a persistent model variant; the benchmark
  tool applies parameters per request and creates nothing.
- Adopt a local weights file: ollama_add_model, which imports it in the
  background under a minimal status badge and returns a progress_id.
- One file from the web: download_file, the primary download tool.
- Several files as one unit of work: the session ceremony below.
- Revisit a past benchmark: benchmark_list_history then
  benchmark_get_saved_result. Diagnose a failure: logs_read for anything
  MSHCore did, ollama_list_logs then ollama_read_log when Ollama itself is
  at fault.

## Single-call tools versus the download queue

Every tool other than the download session tools finishes its whole job in
one call. The download session tools are stateful and split one job across
several calls; nothing is transferred until the queue is started:

    download_create_session       create an empty named queue
    download_add, download_add_many
                                  enqueue URLs only, transfer nothing
    download_start                begin the transfer

Skipping the start leaves a queue that never runs. download_file performs all
three steps itself, so never pair it with them.

## Three kinds of identifier — never interchange them

- session_id names a download queue. You choose it when calling
  download_create_session. Every download_* tool takes it, and only those
  tools accept it. Format: whatever string you picked, or 'auto-<8 hex>' when
  download_file generated one.
- progress_id names one background operation. A tracked tool mints it and
  returns it; you never choose it. Only progress_get_status,
  benchmark_get_result, progress_cancel and progress_pause accept it. Format:
  'download-<date>-<time>-<8 hex>', 'benchmark-<date>-<time>-<8 hex>' or
  'importmodel-<date>-<time>-<8 hex>'.
- benchmark_id names one run kept in the benchmark history. benchmark_list_history
  lists them; benchmark_get_saved_result and benchmark_delete_history take
  one. Format: '<date>T<time>_<6 hex>'. A history id and a progress_id are
  never interchangeable.

download_file returns both, each under its own name — 'progress_id'
and 'session_id'. Passing a session_id to progress_get_status reports
found=false; passing a progress_id to a download_* tool reports an unknown
session. Never call the starting tool again in order to obtain an id — the
first call already returned it, and a second call starts a second operation.

## Long-running operations and progress

Immediate: every read-only tool, and every download_* queue-control tool.

Blocking until finished, with no progress reporting: ollama_run_model,
ollama_start, ollama_stop, ollama_install, python_run_script,
python_install_packages, python_create_environment and python_install_interpreter.
Only ollama_start and ollama_stop take a timeout argument (15 and 10 seconds
by default); the rest have none, several take minutes, and installers have no
progress variant at all. A model import is deliberately not on this list:
ollama_add_model is tracked and reports a progress bar, so it runs in the
background instead.

Background, returning a progress_id at once: download_file, download_start,
benchmark_run and ollama_add_model.
Track them with progress_get_status(progress_id), which is a fast,
non-blocking read of a recorded snapshot: status is 'starting' or 'running'
while work continues, then 'completed', 'failed' or 'cancelled'. It answers
after the operation ends and after this server restarts. An unknown id
reports found=false and is never answered with another operation's progress.

All of them run without you, and every tracked operation follows the same
behaviour: after starting one, say that it has started, then end the turn
and go to sleep. Never hold the turn open polling in a loop — minutes
of polling stall the conversation, the operation runs to completion without
you either way, and its progress bar keeps updating itself. When the user
calls again, wake up, call progress_get_status once with the progress_id,
and take it from there.

The kinds differ in what waking up means:

- Downloads have nothing to collect: starting the transfer is the whole
  job, and the poll only confirms the outcome.
- Model imports are the same: the import is the whole job. A 'completed'
  status means the model is registered and usable — nothing to fetch, only
  ollama_list_models to confirm it if wanted. A failed or cancelled import
  registers nothing under the name.
- Benchmarks return an acknowledgement containing no measurements. On
  waking, call progress_get_status with the same id until the status is
  terminal, then call benchmark_get_result(progress_id) with it: that reads
  the measurements back from the benchmark history the finished run was
  saved into, and it is the only place they exist. Until then, do not
  describe timings, compare configurations or recommend settings — and a
  failed or cancelled run produces no measurements and says so. The run
  stays in the history afterwards: a completed snapshot carries its
  'benchmark_id', and benchmark_list_history and benchmark_get_saved_result
  reach it again in any later conversation. Every background comparison also
  carries a two-way 'significance' assessment: 'by_model' judges each model's
  configurations against each other, 'across_models' judges the models against
  each other. A verdict's 'significant' is a real true/false when repetitions
  ran above 1, null when each prompt was measured once.

## Controlling a running operation

Two controls take a progress_id:

- progress_cancel ends the operation and has MSHCore undo it — partial and
  completed downloads are deleted, a loaded model is unloaded, an import's
  'ollama create' process is killed so nothing registers under the name —
  and records a 'cancelled' entry that logs_read will show. Applies to
  downloads, benchmarks and model imports. Cannot be undone.
- progress_pause suspends a download and keeps the queue, the fetched files
  and the partial data; calling it again resumes. Downloads only; a
  benchmark or a model import reports pause_action='unavailable'.

Cancelling a download also removes its session: the id becomes free, and
downloading the same files again means calling download_create_session and
download_add again. A cancelled session refuses further use. Starting a
session that is already downloading is rejected rather than started twice —
end the turn and go to sleep; when the user calls again, poll the
progress_id named in the error, or cancel it first.

## Sequence rules

- Recommend a model only after reading the hardware: system_get_gpu_info for
  VRAM, system_get_storage_info for free space. Do not infer either.
- Call ollama_get_status before any other Ollama tool, and ollama_start when
  it reports running=false. Every model tool needs the service up.
- Confirm a model is installed with ollama_list_models before benchmarking,
  configuring or running it; the tools fail on an unknown name.
- Never call python_edit_script or python_create_script without first reading
  the target with python_read_script when it may already exist. Both write the
  whole file, and edit discards the previous content unrecoverably.
- Check download_list_allowed_domains before queueing a URL from an unfamiliar
  host. A rejected domain raises before anything is queued.
- On any failure, read logs_get_file_info, then logs_read with line_count set
  and level='ERROR' — a tool's error message is often shorter than the log
  entry behind it. For an Ollama service, model-load or GPU-detection failure,
  call ollama_list_logs and then ollama_read_log with a line range.

Tools that delete models, environments, script files, packages, downloaded
files, or saved benchmark runs are irreversible and annotated destructive.
Verify the target — with
ollama_list_models, python_list_packages, python_read_script,
download_get_status or benchmark_list_history — before calling them.
"""

READ_ONLY = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)

CallableT = TypeVar("CallableT", bound=Callable[..., Any])


def surface_core_errors(function: CallableT) -> CallableT:
    """Forward exceptions raised by MSHCore to the MCP client verbatim.

    The SDK treats any exception other than ``ToolError`` as a crash: the client
    gets a generic ``Error executing tool <name>`` and the real message stays on
    the server. MSHCore raises descriptive exceptions, so they are re-raised here
    with type name and message intact and the original chained as ``__cause__``.
    MSHCore exceptions are forwarded, never replaced.

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
            # Already an MCP-level error, including argument validation failures.
            raise
        except Exception as error:
            raise ToolError(f"{type(error).__name__}: {error}") from error

    return wrapper  # type: ignore[return-value]


# ============================================================
# System — MSHCore.system
# ============================================================

def register_system_tools(server: MCPServer) -> None:
    """Register hardware discovery tools.

    Args:
        server: Server instance the tools are attached to.
    """

    @server.tool(
        name="system_scan",
        title="Scan system hardware",
        description=(
            "Collect the entire machine profile in one call. Secondary tool: "
            "use it only when several categories are needed at once, and "
            "prefer system_get_gpu_info, system_get_storage_info, "
            "system_get_memory_info or system_get_cuda_version for a single "
            "metric. No prerequisites. Read-only; writes its execution-log "
            "entries and changes nothing else. Shells out eight times on "
            "Windows (six PowerShell queries, two nvidia-smi), each with its "
            "own five-second timeout, so it can block for tens of seconds. "
            "Returns one dict with exactly these keys: 'system' (OS name, "
            "version, build, architecture, Python version), 'cpu' (model, "
            "cores, threads, clocks, instruction-set 'features'), 'memory' "
            "('total_gb', 'available_gb', 'used_gb', 'usage_percent', "
            "physical 'modules', 'channels'), 'gpu' ('count', 'devices', "
            "'cuda_version'), and 'storage' (one entry per drive with "
            "'total_gb', 'used_gb', 'free_gb'). Memory and storage figures "
            "here are GiB rounded to two decimals, not bytes — the narrow "
            "tools return raw bytes instead. The response is a fixed size "
            "with no pagination; it grows only with the number of drives, RAM "
            "modules and GPUs."
        ),
        annotations=READ_ONLY,
    )
    @surface_core_errors
    def system_scan() -> dict:
        return scanner.scan_system()

    @server.tool(
        name="system_get_memory_info",
        title="Get RAM usage",
        description=(
            "Report current system RAM usage. Primary tool for a memory "
            "check; system_scan also returns this but costs subprocess calls. "
            "No prerequisites. Reads in-process counters, so it makes no "
            "subprocess call, returns immediately, and mutates nothing. "
            "Returns one dict with exactly 'total', 'available' and 'used' as "
            "integer bytes plus 'usage_percent' as a float from 0 to 100. Use "
            "'available' rather than 'total' when judging whether a model "
            "will fit in RAM. Fixed-size response; no pagination."
        ),
        annotations=READ_ONLY,
    )
    @surface_core_errors
    def system_get_memory_info() -> dict:
        return hardware.get_memory_info()

    @server.tool(
        name="system_get_storage_info",
        title="Get storage capacity",
        description=(
            "List free and used space per mounted drive. Primary tool for "
            "deciding whether a model download or install will fit. No "
            "prerequisites. Makes no subprocess call, returns immediately, "
            "and mutates nothing. Returns a list of dicts, each with exactly "
            "'drive' (for example 'C:\\' on Windows, '/' elsewhere), 'total', "
            "'used' and 'free' as integer bytes. On Windows every existing "
            "drive letter is enumerated in alphabetical order; a drive that "
            "cannot be read is omitted rather than reported as an error. "
            "Returns an empty list when nothing is readable. Response length "
            "equals the number of drives; no pagination."
        ),
        annotations=READ_ONLY,
    )
    @surface_core_errors
    def system_get_storage_info() -> list[dict]:
        return hardware.get_storage_info()

    @server.tool(
        name="system_get_gpu_info",
        title="Get NVIDIA GPU info",
        description=(
            "List the NVIDIA GPUs and their VRAM. Primary tool for sizing a "
            "model against available VRAM. No prerequisites. Runs nvidia-smi "
            "once with a five-second timeout; read-only. Returns a list of "
            "dicts, each with exactly 'name', 'driver', 'vram_total', "
            "'vram_used', 'vram_free' and 'compute_capability'. The three "
            "VRAM values are strings already suffixed with the unit, for "
            "example '24564 MB' — parse the number out before comparing, and "
            "do not treat them as bytes. Returns an empty list, not an error, "
            "when there is no NVIDIA GPU, nvidia-smi is not installed, or the "
            "query times out; an empty list therefore does not prove the "
            "machine has no GPU. Reports AMD and Intel GPUs not at all. "
            "Response length equals the GPU count; no pagination."
        ),
        annotations=READ_ONLY,
    )
    @surface_core_errors
    def system_get_gpu_info() -> list[dict]:
        return hardware.get_nvidia_info()

    @server.tool(
        name="system_get_cuda_version",
        title="Get CUDA version",
        description=(
            "Report the CUDA version the installed NVIDIA driver advertises, "
            "for checking whether a CUDA-dependent runtime is supported. No "
            "prerequisites. Runs nvidia-smi once with a five-second timeout — "
            "a separate invocation from system_get_gpu_info, so call "
            "system_scan instead when both are needed. Read-only. Returns the "
            "version as a string such as '12.2', or null when no NVIDIA "
            "driver is present or the version cannot be parsed; null is never "
            "an error. This is the driver's CUDA capability, not an installed "
            "CUDA toolkit."
        ),
        annotations=READ_ONLY,
    )
    @surface_core_errors
    def system_get_cuda_version() -> str | None:
        return hardware.get_cuda_version()


# ============================================================
# Ollama runtime — MSHCore.ollama.runtime
# ============================================================

def register_ollama_runtime_tools(server: MCPServer) -> None:
    """Register Ollama service lifecycle tools.

    The mutating tools return ``get_status()`` afterwards, since the underlying
    MSHCore functions return ``None``.

    Args:
        server: Server instance the tools are attached to.
    """

    @server.tool(
        name="ollama_get_status",
        title="Get Ollama status",
        description=(
            "Report whether Ollama is usable. Mandatory first step before any "
            "other ollama_* tool: every model, benchmark and runtime tool "
            "assumes the service is up, and none of them checks. Read-only. "
            "Returns one dict with exactly three keys: 'installed' (the "
            "ollama binary is on PATH), 'running' (its HTTP API on "
            "127.0.0.1:11434 answered), and 'version' (string, or null when "
            "the binary did not report one — null does not mean not "
            "installed). When 'installed' is false, no other Ollama tool can "
            "succeed: run ollama_install first. When 'installed' is true and "
            "'running' is false, call ollama_start. Blocks up to about seven "
            "seconds while probing the API and reading the version."
        ),
        annotations=READ_ONLY,
    )
    @surface_core_errors
    def ollama_get_status() -> dict:
        return runtime.get_status()

    @server.tool(
        name="ollama_list_logs",
        title="List Ollama log files",
        description=(
            "Index Ollama's own log files without reading their contents: the "
            "live app.log and server.log plus rotated app-N.log and "
            "server-N.log copies, from %LOCALAPPDATA%\\Ollama on Windows or "
            "~/.ollama/logs and /var/log/ollama elsewhere. Mandatory first "
            "step before ollama_read_log, because which names exist depends "
            "on how often Ollama has rotated its logs. Read-only. Returns one "
            "dict with exactly 'files' (list of {name, path, size_bytes, "
            "line_count, modified}, most recently modified first), "
            "'directories' (those actually searched) and 'total_bytes'. Use "
            "'size_bytes' and 'line_count' to plan pagination: "
            "ollama_read_log returns an entire file unless given a line "
            "range, and server.log is often megabytes. 'line_count' is "
            "counted exactly as ollama_read_log numbers lines, so it can be "
            "passed straight back as end_line. Every file is read once to "
            "count lines, so this blocks in proportion to total log size. "
            "Fails when no Ollama log directory exists; returns an empty "
            "'files' list when the directories exist but hold no logs."
        ),
        annotations=READ_ONLY,
    )
    @surface_core_errors
    def ollama_list_logs() -> dict:
        return runtime.list_ollama_logs()

    @server.tool(
        name="ollama_read_log",
        title="Read an Ollama log file",
        description=(
            "Read the contents of one Ollama log file. This is the diagnostic "
            "of last resort for Ollama itself — why the service will not "
            "start, why a model failed to load, why the GPU was not detected "
            "— and it is server.log that records all three; app.log covers "
            "the desktop application. These are Ollama's logs; logs_read "
            "serves this project's own execution log instead. Requires a "
            "prior ollama_list_logs call: file_name must be one bare name "
            "from its 'files' list, matched exactly and case-sensitively, "
            "never a path and never guessed. Read-only. PAGINATE: with no "
            "range the whole file is returned untruncated and server.log can "
            "be megabytes, which will overflow the context — pass start_line "
            "and end_line (1-based, inclusive on both ends, so 1..500 is 500 "
            "lines) sized against the 'line_count' ollama_list_logs reported, "
            "and prefer the smallest file covering the period of interest. "
            "The response carries 'total_lines' for the whole file plus the "
            "'start_line' and 'end_line' actually returned; 'end_line' is the "
            "real last line read, so an end_line past the end of the file "
            "comes back as the true final line number. Both are null with "
            "empty 'content' when the range began past the end of the file — "
            "that is not an error. Fails when the name is not in the index."
        ),
        annotations=READ_ONLY,
    )
    @surface_core_errors
    def ollama_read_log(
        file_name: str,
        start_line: int = 1,
        end_line: int | None = None,
    ) -> dict:
        """Read one Ollama log file, in full or a line range.

        Args:
            file_name: Log file name from ollama_list_logs, for example
                'server.log' or 'app-2.log'.
            start_line: First line to return, 1-based. Defaults to the first.
            end_line: Last line to return, inclusive. Optional.

        Returns:
            dict: The file's name, path, size, modification time, total line
                count, the range actually returned, and its contents.
        """
        return runtime.read_ollama_logs(
            file_name=file_name,
            start_line=start_line,
            end_line=end_line,
        )

    @server.tool(
        name="ollama_start",
        title="Start Ollama service",
        description=(
            "Start the Ollama background server and block until its HTTP API "
            "answers. Primary tool for bringing the service up; call it when "
            "ollama_get_status reports installed=true and running=false. "
            "Requires Ollama to be installed — it launches the binary, it "
            "does not install it. Spawns a detached 'ollama serve' process "
            "that outlives this call; does nothing when the API is already "
            "answering, so it is safe to repeat. Blocks up to the timeout "
            "argument, 15 seconds by default. Fails when Ollama is not "
            "installed or the API does not answer within the timeout. Returns "
            "the same 'installed'/'running'/'version' dict as "
            "ollama_get_status, read after the attempt."
        ),
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        ),
    )
    @surface_core_errors
    def ollama_start(timeout: float = runtime.START_TIMEOUT) -> dict:
        """Start the Ollama service.

        Args:
            timeout: Seconds to wait for the API to become ready.

        Returns:
            dict: Runtime status after the start attempt.
        """
        runtime.start(timeout=timeout)
        return runtime.get_status()

    @server.tool(
        name="ollama_stop",
        title="Stop Ollama service",
        description=(
            "Terminate the Ollama server process and block until its API "
            "stops answering. Destructive and machine-wide: on Windows it "
            "force-kills every ollama.exe rather than shutting down "
            "gracefully, so any in-flight generation is lost and every model "
            "resident in memory is unloaded. Do not call it while a benchmark "
            "or a model run is outstanding. Does nothing when Ollama is not "
            "installed or is already stopped. Blocks up to the timeout "
            "argument, 10 seconds by default, and fails if the API is still "
            "answering after it. Returns the same "
            "'installed'/'running'/'version' dict as ollama_get_status, read "
            "after the attempt. Models stay installed on disk; use "
            "ollama_stop_model to free one model's VRAM without stopping the "
            "service."
        ),
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=True,
            idempotent_hint=True,
            open_world_hint=False,
        ),
    )
    @surface_core_errors
    def ollama_stop(timeout: float = runtime.STOP_TIMEOUT) -> dict:
        """Stop the Ollama service.

        Args:
            timeout: Seconds to wait for shutdown.

        Returns:
            dict: Runtime status after the stop attempt.
        """
        runtime.stop(timeout=timeout)
        return runtime.get_status()

    @server.tool(
        name="ollama_install",
        title="Install Ollama",
        description=(
            "Install Ollama by executing an installer already present on "
            "disk. installer_path must be a local filesystem path to that "
            "executable, not a URL and not a download identifier — this tool "
            "downloads nothing. Obtain the installer first with download_file "
            "from an allowed domain, and pass the 'destination' it reported. "
            "Modifies system state outside this project. No silent flag is "
            "passed, so the installer may open a window and wait for input; "
            "it blocks with no timeout, and an interactive prompt will hang "
            "this call until the installer exits. There is no progress "
            "variant, so nothing can be polled while it runs. Fails when the "
            "path is not a file or the installer exits non-zero. Returns the "
            "same 'installed'/'running'/'version' dict as ollama_get_status, "
            "read afterwards; the install is not verified beyond that."
        ),
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=False,
            open_world_hint=False,
        ),
    )
    @surface_core_errors
    def ollama_install(installer_path: str) -> dict:
        """Install Ollama from a local installer file.

        Args:
            installer_path: Path to the installer executable on disk.

        Returns:
            dict: Runtime status after installation.
        """
        runtime.install(installer_path=installer_path)
        return runtime.get_status()


# ============================================================
# Ollama models — MSHCore.ollama.model
# ============================================================

def register_ollama_model_tools(server: MCPServer) -> None:
    """Register Ollama model management tools.

    Args:
        server: Server instance the tools are attached to.
    """

    @server.tool(
        name="ollama_list_models",
        title="List installed models",
        description=(
            "List the models installed on this machine. Primary tool for "
            "discovering what can be run, and the way to confirm a model name "
            "before ollama_run_model, ollama_show_model_info, "
            "ollama_configure_model, ollama_remove_model or any benchmark — "
            "those all fail on an unknown name. Requires the Ollama service "
            "to be running (ollama_get_status). Read-only. Returns the raw "
            "text table printed by 'ollama list', with NAME, ID, SIZE and "
            "MODIFIED columns, not a parsed structure; parse the NAME column "
            "for exact tags. An empty string means either no models are "
            "installed or the command failed — this tool does not distinguish "
            "the two, so check ollama_get_status when the result is "
            "unexpectedly empty. Response grows one line per installed model; "
            "no pagination."
        ),
        annotations=READ_ONLY,
    )
    @surface_core_errors
    def ollama_list_models() -> str:
        return model.list_models()

    @server.tool(
        name="ollama_list_running_models",
        title="List running models",
        description=(
            "List the models currently resident in memory, as opposed to "
            "installed on disk. Use it to see what is occupying VRAM before "
            "loading another model or starting a benchmark; use "
            "ollama_list_models for what is installed. Requires the Ollama "
            "service to be running. Read-only. Returns the raw text table "
            "printed by 'ollama ps', including each model's size and "
            "keep-alive expiry, not a parsed structure. An empty string means "
            "no model is loaded or the command failed; the two are not "
            "distinguished. Response grows one line per loaded model; no "
            "pagination."
        ),
        annotations=READ_ONLY,
    )
    @surface_core_errors
    def ollama_list_running_models() -> str:
        return model.list_running_models()

    @server.tool(
        name="ollama_show_model_info",
        title="Show model details",
        description=(
            "Report one installed model's architecture, parameter count, "
            "quantization, context length and baked-in parameters. Primary "
            "tool for checking whether a model fits the hardware and what "
            "context length it supports, before running or benchmarking it. "
            "Requires the Ollama service to be running and the model to be "
            "installed: model_name is a name or tag exactly as "
            "ollama_list_models prints it in its NAME column, for example "
            "'llama3' or 'llama3:8b'. Read-only. Returns the raw text 'ollama "
            "show' prints, not a parsed structure. Fails when the model is "
            "not installed. Fixed-size response; no pagination."
        ),
        annotations=READ_ONLY,
    )
    @surface_core_errors
    def ollama_show_model_info(model_name: str) -> str:
        """Show details for an installed model.

        Args:
            model_name: Model name or tag, for example 'llama3' or 'llama3:8b'.

        Returns:
            str: Model metadata as reported by 'ollama show'.
        """
        return model.show_model_info(model=model_name)

    @server.tool(
        name="ollama_run_model",
        title="Run a prompt",
        description=(
            "Send one prompt to a model and return the text it generates. "
            "Primary tool for getting output from a local model. Requires the "
            "Ollama service to be running and model_name to be installed, as "
            "printed by ollama_list_models. Single-shot: no conversation "
            "history is kept, so each call is independent, and generation "
            "parameters cannot be set here — use benchmark_run to "
            "vary them, or ollama_configure_model to bake them into a "
            "variant. Returns the generated text only, with no timings or "
            "token counts; use benchmark_run when you need "
            "measurements. Blocks "
            "for the whole generation with no timeout and no progress "
            "reporting, which for a large model or a long prompt can be "
            "minutes."
        ),
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=False,
            open_world_hint=False,
        ),
    )
    @surface_core_errors
    def ollama_run_model(model_name: str, prompt: str) -> str:
        """Run a single prompt against a model.

        Args:
            model_name: Target model name or tag.
            prompt: Prompt text to send.

        Returns:
            str: Generated output text.
        """
        return model.run_model(model=model_name, prompt=prompt)

    @server.tool(
        name="ollama_load_model",
        title="Preload a model",
        description=(
            "Load a model into VRAM or system memory ahead of use, so the "
            "first real prompt does not pay the load cost. Optional "
            "optimisation: the benchmark tools preload on their own, and "
            "ollama_run_model loads implicitly, so this is only worth calling "
            "to warm a model before timing something yourself. This is a "
            "memory-only operation — it installs nothing, downloads nothing, "
            "and the model must already be installed; use download_file plus "
            "ollama_add_model to put a new model on disk. Requires the Ollama "
            "service to be running. The model stays resident for keep_alive "
            "(an Ollama duration string such as '10m' or '1h'), occupying "
            "VRAM until it expires or ollama_stop_model unloads it. Returns "
            "Ollama's load response dict, or null when the model was already "
            "resident — null means nothing needed doing, not a failure. "
            "Blocks up to 300 seconds while the weights are read."
        ),
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        ),
    )
    @surface_core_errors
    def ollama_load_model(
        model_name: str,
        keep_alive: str = "10m",
    ) -> dict | None:
        """Preload a model into memory.

        Args:
            model_name: Model name or tag to load.
            keep_alive: How long to keep it resident, for example '10m' or '1h'.

        Returns:
            dict | None: Ollama load response, or null if already loaded.
        """
        return model.load_model(model=model_name, keep_alive=keep_alive)

    @server.tool(
        name="ollama_stop_model",
        title="Unload a model",
        description=(
            "Unload one model from memory and free its VRAM, leaving it "
            "installed on disk and leaving the Ollama service running. "
            "Primary tool for reclaiming VRAM before loading a larger model; "
            "ollama_stop kills the whole service instead, and "
            "ollama_remove_model deletes the model permanently. Requires the "
            "Ollama service to be running. model_name must name a model "
            "currently loaded, as printed by ollama_list_running_models. "
            "Nothing on disk is deleted and no data is lost, but any "
            "generation in flight for that model is interrupted. Returns the "
            "raw text 'ollama stop' prints, usually empty — an empty string "
            "is success, not a failure."
        ),
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        ),
    )
    @surface_core_errors
    def ollama_stop_model(model_name: str) -> str:
        """Unload a running model.

        Args:
            model_name: Running model name to stop.

        Returns:
            str: Output from 'ollama stop'.
        """
        return model.stop_model(model=model_name)

    @server.tool(
        name="ollama_delete_model_file",
        title="Delete a model weights file",
        description=(
            "Delete one model weights file from disk — the .gguf a manual "
            "download or a download_file call produced and an import has "
            "already copied into Ollama's store. DESTRUCTIVE and "
            "IRREVERSIBLE: the file is unlinked immediately, not moved to a "
            "recycle bin, and there is no backup. model_path is a local "
            "filesystem path, typically the 'destination' download_file "
            "reported; only files ending in '.gguf' are accepted, so a "
            "mistyped path fails rather than removing anything else. "
            "Deleting the file does not touch Ollama: a model already "
            "registered from it keeps working, since the import copied the "
            "weights — delete the registered copy with ollama_remove_model "
            "instead, and neither undoes the other. Confirm the path with "
            "the user before calling when the model was not imported in "
            "this conversation. Returns the deleted file's absolute path. "
            "Fails when the path is not a .gguf file or does not exist."
        ),
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=True,
            idempotent_hint=True,
            open_world_hint=False,
        ),
    )
    @surface_core_errors
    def ollama_delete_model_file(model_path: str) -> str:
        """Delete a GGUF model weights file from disk.

        Args:
            model_path: Path to the model file on disk.

        Returns:
            str: Absolute path of the deleted file.
        """
        return model.delete_model_file(model_path=model_path)

    @server.tool(
        name="ollama_configure_model",
        title="Create a configured model variant",
        description=(
            "Create a new, separately named model from an existing one with "
            "Modelfile PARAMETER values baked in, for example temperature or "
            "num_ctx. Use this only when a persistent variant is wanted — for "
            "trying parameters out, benchmark_run "
            "applies them per request and creates nothing. Requires the Ollama "
            "service to be running and source_model to be installed, as "
            "printed by ollama_list_models. source_model is left completely "
            "unchanged. target_model is the new name: an existing model of "
            "that name is overwritten with no warning and no way back, so "
            "verify it is free first. Each variant is a new entry in Ollama's "
            "store, so repeated calls accumulate models — remove them with "
            "ollama_remove_model. parameters is a flat dict of PARAMETER "
            "key-values, for example {'temperature': 0.7, 'num_ctx': 4096}; "
            "entries whose value is null are dropped, and at least one must "
            "survive — an empty dict, or one whose every value is null, "
            "fails. An invalid key or value is only detected by Ollama. "
            "Blocks until Ollama finishes creating the variant. Returns the "
            "raw text 'ollama create' prints."
        ),
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=False,
            open_world_hint=False,
        ),
    )
    @surface_core_errors
    def ollama_configure_model(
        source_model: str,
        target_model: str,
        parameters: dict,
    ) -> str:
        """Create a configured variant of an existing model.

        Args:
            source_model: Existing model to base the variant on.
            target_model: Name for the new model.
            parameters: Modelfile PARAMETER key-values, for example
                {"temperature": 0.7, "num_ctx": 4096}.

        Returns:
            str: Output from 'ollama create'.
        """
        return model.configure_model(
            source_model=source_model,
            target_model=target_model,
            parameters=parameters,
        )

    @server.tool(
        name="ollama_remove_model",
        title="Delete a model",
        description=(
            "Permanently delete a model from Ollama's local storage, freeing "
            "its disk space. IRREVERSIBLE: there is no recycle bin and no "
            "undo — recovering the model means downloading or re-importing "
            "it. Confirm the exact name with ollama_list_models before "
            "calling, since a tag such as 'llama3' and 'llama3:8b' are "
            "different entries, and deleting a source model breaks nothing "
            "about a variant made from it but cannot be reversed. Requires "
            "the Ollama service to be running. To free VRAM without deleting "
            "anything use ollama_stop_model instead. Returns the raw text "
            "'ollama rm' prints, normally \"deleted '<model>'\". Fails when "
            "the model is not installed."
        ),
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=True,
            idempotent_hint=False,
            open_world_hint=False,
        ),
    )
    @surface_core_errors
    def ollama_remove_model(model_name: str) -> str:
        """Delete a model from local storage.

        Args:
            model_name: Model name to remove.

        Returns:
            str: Output from 'ollama rm'.
        """
        return model.remove_model(model=model_name)


# ============================================================
# Benchmarking — MSHCore.benchmark (runner and history)
# ============================================================

def register_benchmark_tools(server: MCPServer) -> None:
    """Register benchmark history tools.

    The benchmark itself — ``benchmark_run`` — is the tracked tool
    bound to the progress panel in ``gui.app``; only the history it saves
    its completed runs into is registered here as plain tools.

    Args:
        server: Server instance the tools are attached to.
    """

    @server.tool(
        name="benchmark_list_history",
        title="List saved benchmark runs",
        description=(
            "List every benchmark run kept in the history, newest first. "
            "Every completed background benchmark — benchmark_run — "
            "is saved here automatically, as is nothing else: the starting "
            "tool returns only a progress_id, no measurements. "
            "Read-only and cheap: the listing comes from a small index file, "
            "not the results themselves. Call it to revisit a comparison "
            "made earlier, to pick an id for benchmark_get_saved_result, or "
            "to see whether a progress_id's run is already in the history. "
            "Returns one record per run with 'id' (the benchmark_id; this — "
            "not a progress_id — is what benchmark_get_saved_result and "
            "benchmark_delete_history take), 'saved_at', 'model' or "
            "'models', 'repetitions', "
            "'configuration_count', 'prompt_count', 'winner' (the fastest "
            "pair's label — the configuration's name, prefixed with the "
            "model's name once several models ran) and 'significant'. The "
            "history "
            "keeps the most recent runs only, so old ids disappear as new "
            "runs arrive."
        ),
        annotations=READ_ONLY,
    )
    @surface_core_errors
    def benchmark_list_history() -> list[dict]:
        """List the saved benchmark runs, newest first.

        Returns:
            list[dict]: One record per saved run.
        """
        return history.list_saved()

    @server.tool(
        name="benchmark_get_saved_result",
        title="Read a saved benchmark run",
        description=(
            "Read one benchmark run from the history in full, by the "
            "benchmark_id benchmark_list_history reported. Returns the "
            "header — 'id', "
            "'saved_at', 'model' or 'models', 'prompts', 'configurations', "
            "'repetitions', 'summary' with the winner and significance "
            "verdict — plus the complete comparison result under 'result': "
            "per-prompt timings, token counts, per-prompt ttft, VRAM and "
            "GPU readings, the run-to-run spreads, and the 'significance' "
            "assessment. This is the read tool for runs made in earlier "
            "conversations or before this server restarted; for a run "
            "started in this conversation you more likely want "
            "benchmark_get_result with its progress_id instead, which also "
            "works for runs still in flight. Read-only."
        ),
        annotations=READ_ONLY,
    )
    @surface_core_errors
    def benchmark_get_saved_result(benchmark_id: str) -> dict:
        """Read one saved benchmark run in full.

        Args:
            benchmark_id: Identifier from benchmark_list_history.

        Returns:
            dict: The stored record — header, summary and full result.
        """
        return history.load(benchmark_id)

    @server.tool(
        name="benchmark_delete_history",
        title="Delete a saved benchmark run",
        description=(
            "Permanently remove one benchmark run from the history, by the "
            "benchmark_id benchmark_list_history reported. DESTRUCTIVE and "
            "IRREVERSIBLE: the result file and its index record are gone — "
            "read the run with benchmark_get_saved_result first if its "
            "measurements might still matter. The result files are a few "
            "hundred kilobytes each and the history caps itself, so "
            "deleting is for wiping a run whose numbers were wrong, not "
            "for freeing space. Takes a benchmark_id, never a progress_id. "
            "Returns true when a run was removed, false when nothing was "
            "stored under that id — deleting a missing id is the state "
            "asked for, not a failure."
        ),
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=True,
            idempotent_hint=True,
            open_world_hint=False,
        ),
    )
    @surface_core_errors
    def benchmark_delete_history(benchmark_id: str) -> bool:
        """Remove one saved benchmark run from the history.

        Args:
            benchmark_id: Identifier from benchmark_list_history.

        Returns:
            bool: True when a run was removed, False when there was none.
        """
        return history.delete(benchmark_id)


# ============================================================
# Python — MSHCore.python
# ============================================================

def register_python_tools(server: MCPServer) -> None:
    """Register Python interpreter, environment, package, and script tools.

    Args:
        server: Server instance the tools are attached to.
    """

    @server.tool(
        name="python_list_versions",
        title="List installed Python versions",
        description=(
            "List the Python interpreters installed on this machine. Primary "
            "tool for discovering which versions are available before "
            "creating an environment or deciding whether "
            "python_install_interpreter is needed. No prerequisites. Read-only; "
            "reads the Windows registry under HKLM and HKCU, so it makes no "
            "subprocess call and returns immediately. Returns a list of "
            "dicts, each with exactly 'version' and 'path' (the absolute "
            "interpreter path). The interpreter running this server is always "
            "included, so the list is never empty. Version granularity is "
            "inconsistent: registry entries report major.minor such as "
            "'3.12', while the running interpreter reports full "
            "major.minor.micro. A version present in both registry hives "
            "appears twice. Response length equals the interpreter count; no "
            "pagination."
        ),
        annotations=READ_ONLY,
    )
    @surface_core_errors
    def python_list_versions() -> list[dict]:
        return installer.get_python_status()

    @server.tool(
        name="python_resolve_path",
        title="Resolve interpreter path",
        description=(
            "Resolve the absolute path of the interpreter a given environment "
            "would use. Secondary tool: every python_* tool resolves this "
            "internally, so call it only to report the path or to verify an "
            "environment before using it. env_path is a virtual environment "
            "directory as passed to or returned by python_create_environment; "
            "omit it to get the interpreter running this server. Read-only; "
            "no subprocess, returns immediately. Returns the path as a "
            "string. Fails when env_path is given but holds no interpreter — "
            "which makes it a cheap existence check for an environment. It "
            "does not otherwise verify that the directory is a valid virtual "
            "environment."
        ),
        annotations=READ_ONLY,
    )
    @surface_core_errors
    def python_resolve_path(env_path: str | None = None) -> str:
        """Resolve an interpreter path.

        Args:
            env_path: Optional virtual environment directory.

        Returns:
            str: Absolute path to the interpreter executable.
        """
        return environment.get_python_path(environment=env_path)

    @server.tool(
        name="python_create_environment",
        title="Create a virtual environment",
        description=(
            "Create a new virtual environment on disk. Primary tool for "
            "isolating packages before python_install_packages or "
            "python_run_script; pass the same path to those tools afterwards "
            "as their env_path. No prerequisites. env_path is the directory "
            "to create — it must not already exist, and the call fails rather "
            "than overwriting anything, so an existing environment is never "
            "disturbed. The environment always uses the interpreter running "
            "this server; there is no way to select a version here, so check "
            "python_list_versions first if a specific version matters. Blocks "
            "while venv runs, with no timeout and no progress reporting. "
            "Returns the created environment's absolute path as a string, "
            "which is the value to reuse as env_path."
        ),
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=False,
            open_world_hint=False,
        ),
    )
    @surface_core_errors
    def python_create_environment(env_path: str) -> str:
        """Create a virtual environment.

        Args:
            env_path: Directory to create the environment in.

        Returns:
            str: Absolute path to the created environment.
        """
        return environment.create_environment(path=env_path)

    @server.tool(
        name="python_remove_environment",
        title="Delete a virtual environment",
        description=(
            "Recursively delete a directory and everything inside it, "
            "including every package installed there. IRREVERSIBLE: the tree "
            "is unlinked with no backup and no recycle bin. It does NOT "
            "verify that the target is a virtual environment — any existing "
            "directory path given here is destroyed, so confirm env_path is "
            "the environment you mean, ideally by resolving it first with "
            "python_resolve_path. env_path is the environment directory as "
            "returned by python_create_environment. Fails when the path does "
            "not exist. A failure part-way through, for example a locked file "
            "on Windows, can leave the tree partly deleted. Blocks for the "
            "duration of the walk. Returns a confirmation string naming the "
            "removed path."
        ),
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=True,
            idempotent_hint=False,
            open_world_hint=False,
        ),
    )
    @surface_core_errors
    def python_remove_environment(env_path: str) -> str:
        """Delete a virtual environment.

        Args:
            env_path: Environment directory to remove.

        Returns:
            str: Confirmation message naming the removed path.
        """
        environment.remove_environment(path=env_path)
        return f"Environment removed: {env_path}"

    @server.tool(
        name="python_list_packages",
        title="List installed packages",
        description=(
            "List the packages installed for one interpreter, with their "
            "versions. Primary tool for checking whether a dependency is "
            "present and at what version, and the way to confirm a package "
            "name before python_uninstall_packages. env_path is a virtual "
            "environment directory as returned by python_create_environment; "
            "omit it to inspect the interpreter running this server. "
            "Read-only. Returns the raw text table 'pip list' prints, not a "
            "parsed structure — parse the two columns yourself. Blocks while "
            "pip runs, with no timeout. Response grows one line per installed "
            "package and there is no pagination or filter, so an environment "
            "with hundreds of packages produces a long result."
        ),
        annotations=READ_ONLY,
    )
    @surface_core_errors
    def python_list_packages(env_path: str | None = None) -> str:
        """List installed packages.

        Args:
            env_path: Optional virtual environment to inspect.

        Returns:
            str: Output from 'pip list'.
        """
        return tools.list_packages(environment=env_path)

    @server.tool(
        name="python_install_packages",
        title="Install packages",
        description=(
            "Install packages with pip into one interpreter. Primary tool for "
            "adding dependencies. env_path is a virtual environment directory "
            "as returned by python_create_environment; omitting it installs "
            "into the interpreter running this server, which mutates the "
            "environment this server itself depends on — prefer an explicit "
            "environment. packages is a non-empty list of requirement "
            "strings, each optionally pinned, for example 'numpy==1.26.4'; "
            "pin versions when reproducibility matters, and an empty list "
            "fails. Reaches the network and downloads "
            "from the configured package index, and installing a package can "
            "upgrade or downgrade its dependencies in place. Blocks until pip "
            "exits, with no timeout and no progress reporting, which for "
            "large wheels can be minutes. Returns pip's stdout as raw text; "
            "pip's warnings on stderr are discarded on success. Fails on a "
            "non-zero pip exit with pip's stderr as the error. Confirm the "
            "outcome with python_list_packages."
        ),
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=False,
            open_world_hint=True,
        ),
    )
    @surface_core_errors
    def python_install_packages(
        packages: list[str],
        env_path: str | None = None,
    ) -> str:
        """Install packages with pip.

        Args:
            packages: Package names or version specifiers.
            env_path: Optional target virtual environment.

        Returns:
            str: Output from 'pip install'.
        """
        return tools.install_packages(
            packages=packages,
            environment=env_path,
        )

    @server.tool(
        name="python_uninstall_packages",
        title="Uninstall packages",
        description=(
            "Remove packages with pip from one interpreter. DESTRUCTIVE and "
            "unconfirmed: pip runs with -y, so there is no prompt, and a "
            "package's dependents are left broken rather than removed. "
            "Confirm the exact names with python_list_packages first. "
            "packages is the list of package names to remove; at least one "
            "is required. env_path is a virtual environment directory as "
            "returned by "
            "python_create_environment; omitting it targets the interpreter "
            "running this server, and removing a package there can break this "
            "server itself — always pass an environment unless the intent is "
            "explicitly to change the server's own interpreter. Blocks until "
            "pip exits, with no timeout. Returns pip's stdout as raw text. "
            "Naming a package that is not installed is not an error, so "
            "success does not prove anything was removed — verify with "
            "python_list_packages."
        ),
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=True,
            idempotent_hint=False,
            open_world_hint=False,
        ),
    )
    @surface_core_errors
    def python_uninstall_packages(
        packages: list[str],
        env_path: str | None = None,
    ) -> str:
        """Uninstall packages with pip.

        Args:
            packages: Package names to remove.
            env_path: Optional target virtual environment.

        Returns:
            str: Output from 'pip uninstall'.
        """
        return tools.uninstall_packages(
            packages=packages,
            environment=env_path,
        )

    @server.tool(
        name="python_create_script",
        title="Create a script file",
        description=(
            "Write a new Python script to disk. Primary tool for adding a "
            "script; use python_edit_script to change one that already "
            "exists. path must end in .py and is the file to create; missing "
            "parent directories are created for it. Fails if the file already "
            "exists, so no existing content can ever be lost here — on that "
            "failure, read the file with python_read_script and decide "
            "whether to overwrite it with python_edit_script. content is the "
            "complete source text, written as UTF-8. Writes anywhere on the "
            "filesystem the server can reach; there is no sandbox. Returns "
            "the created file's absolute path as a string, which is the value "
            "to pass to python_run_script."
        ),
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=False,
            open_world_hint=False,
        ),
    )
    @surface_core_errors
    def python_create_script(path: str, content: str) -> str:
        """Create a Python script file.

        Args:
            path: Destination file path.
            content: Script source text.

        Returns:
            str: Absolute path to the created script.
        """
        return tools.create_script(path=path, content=content)

    @server.tool(
        name="python_read_script",
        title="Read a script file",
        description=(
            "Return a Python script's full source text. MANDATORY before "
            "python_edit_script: that tool replaces the whole file and the "
            "previous content cannot be recovered, so read it first whenever "
            "any of it must be preserved. Also the way to inspect a script "
            "before python_run_script executes it. path must end in .py and "
            "name an existing file. Read-only. Returns the source exactly as "
            "stored, unmodified and untruncated — the whole file arrives in "
            "the response, with no line range and no pagination, so a very "
            "large script consumes context proportionally. Fails when the "
            "path does not exist or is not a .py file."
        ),
        annotations=READ_ONLY,
    )
    @surface_core_errors
    def python_read_script(path: str) -> str:
        """Read a Python script file.

        Args:
            path: Script file to read.

        Returns:
            str: Source text of the script.
        """
        return tools.read_script(path=path)

    @server.tool(
        name="python_edit_script",
        title="Overwrite a script file",
        description=(
            "Replace a Python script's entire contents. FULL OVERWRITE, NOT A "
            "PATCH: content becomes the whole file, everything previously in "
            "it is discarded, and there is no backup or undo. Requires a "
            "prior python_read_script call on the same path whenever any "
            "existing content must survive — send the complete intended file, "
            "not a fragment. path must end in .py and name an existing file; "
            "it will not create one, so use python_create_script for a new "
            "script. Returns the file's absolute path as a string."
        ),
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=True,
            idempotent_hint=True,
            open_world_hint=False,
        ),
    )
    @surface_core_errors
    def python_edit_script(path: str, content: str) -> str:
        """Overwrite a Python script file.

        Args:
            path: Script file to rewrite.
            content: New script source text.

        Returns:
            str: Absolute path to the updated script.
        """
        return tools.edit_script(path=path, content=content)

    @server.tool(
        name="python_delete_script",
        title="Delete a script file",
        description=(
            "Delete one Python script file from disk. IRREVERSIBLE: the file "
            "is unlinked immediately, not moved to a recycle bin, and there "
            "is no backup. Read it with python_read_script first if its "
            "contents may still be wanted. path must end in .py and name an "
            "existing file; directories are refused, and only that single "
            "file is removed. Returns the deleted file's absolute path as a "
            "string. Fails when the path does not exist."
        ),
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=True,
            idempotent_hint=False,
            open_world_hint=False,
        ),
    )
    @surface_core_errors
    def python_delete_script(path: str) -> str:
        """Delete a Python script file.

        Args:
            path: Script file to remove.

        Returns:
            str: Absolute path of the deleted script.
        """
        return tools.delete_script(path=path)

    @server.tool(
        name="python_run_script",
        title="Run a script",
        description=(
            "Execute a Python script file and return what it printed. "
            "DESTRUCTIVE by delegation: the script runs with this server's "
            "full privileges and can read, write or delete anything the "
            "server can, with no sandbox — read it with python_read_script "
            "before running anything you did not just write. path must end in "
            ".py and name an existing file, for example the path "
            "python_create_script returned. env_path is a virtual environment "
            "directory as returned by python_create_environment; omit it to "
            "use the interpreter running this server. The script starts with "
            "the server's working directory, not its own, and with stdin "
            "closed, so any call to input() raises EOFError and fails the "
            "run. Blocks until the process exits, with no timeout, no "
            "progress reporting and no way to kill it — an infinite loop "
            "hangs this call indefinitely. Returns stdout as raw text, "
            "stripped; stderr is discarded on success. On a non-zero exit it "
            "fails with the script's stderr as the error and stdout is lost, "
            "so have the script print diagnostics to stdout if you need them "
            "either way."
        ),
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=True,
            idempotent_hint=False,
            open_world_hint=True,
        ),
    )
    @surface_core_errors
    def python_run_script(
        path: str,
        env_path: str | None = None,
    ) -> str:
        """Execute a Python script.

        Args:
            path: Script file to run.
            env_path: Optional virtual environment to run it with.

        Returns:
            str: Standard output from the script.
        """
        return tools.run_script(path=path, environment=env_path)

    @server.tool(
        name="python_install_interpreter",
        title="Install Python from an installer",
        description=(
            "Install a Python interpreter by executing a Windows installer "
            "already present on disk. installer_path is a local filesystem "
            "path to that executable, not a URL — this tool downloads "
            "nothing; fetch the installer first with download_file from "
            "python.org and pass the 'destination' it reported. Modifies "
            "system state outside this project: it runs quietly with PATH "
            "prepending enabled, so the machine's default 'python' can "
            "change. all_users=true installs machine-wide and needs "
            "administrator rights, which this tool does not acquire — without "
            "them the installer exits non-zero and the call fails. Blocks "
            "until the installer exits, with no timeout and no progress "
            "variant to poll. Returns the same list of 'version'/'path' dicts "
            "as python_list_versions, re-detected afterwards; compare it against "
            "a python_list_versions call made before the install to confirm a "
            "new version actually appeared, since the installer's own output "
            "is not returned."
        ),
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=False,
            open_world_hint=False,
        ),
    )
    @surface_core_errors
    def python_install_interpreter(
        installer_path: str,
        all_users: bool = False,
    ) -> list[dict]:
        """Install Python from a local installer.

        Args:
            installer_path: Path to the installer executable.
            all_users: Whether to install for all users instead of the current one.

        Returns:
            list[dict]: Detected Python versions and paths after installation.
        """
        return installer.install_python(
            installer_path=installer_path,
            all_users=all_users,
        )


# ============================================================
# Downloads — MSHCore.download_manager
# ============================================================

# How long a cancellation waits for the download worker to stop before the tool
# reports back. The worker exits at a chunk boundary, so this only has to cover
# one chunk read plus MSHCore's own cleanup.
CANCEL_WAIT_SECONDS = 60.0

# DownloadManager is stateful: a queue is built up, started, then controlled and
# polled while a background thread downloads. MCP tool calls are individually
# stateless, so named manager instances are kept here and each tool acts on one
# by session_id. This registry is the only state this layer adds; queueing,
# retrying, resuming, and progress tracking all stay in MSHCore. The registry's
# three operations — look up, add, remove — are what the gui layer's starting
# tools receive, so a session opened by download_file on the panel is the same
# object the plain download tools act on.
#
# A cancelled session is dropped from the registry as part of the cancellation,
# not left behind in a cancelled state: its queue and files are gone, so keeping
# it would let a later download_add append to a queue that still held the
# original URLs and start the same transfer twice.
_sessions: dict[str, DownloadManager] = {}
_sessions_lock = threading.Lock()


def _register_session(session_id: str, manager: DownloadManager) -> None:
    """Add a new session to the registry.

    Args:
        session_id: Identifier for the session, generated or caller-chosen.
        manager: Manager instance the session id names.
    """
    with _sessions_lock:
        _sessions[session_id] = manager


def _get_session(session_id: str) -> DownloadManager:
    """Look up a live download session.

    Args:
        session_id: Identifier passed to ``download_create_session``.

    Returns:
        DownloadManager: The registered manager instance.

    Raises:
        ToolError: If no session is registered under that identifier, or if the
            one registered has been cancelled and is only awaiting removal.
    """
    with _sessions_lock:
        manager = _sessions.get(session_id)
        known = ", ".join(sorted(_sessions)) or "none"

    if manager is None:
        raise ToolError(
            f"Unknown download session: '{session_id}' (open sessions: {known})"
        )

    if manager.get_status()["closed"]:
        # Reached only if a cancellation's own removal has not run yet — a
        # cancelled session is otherwise gone from the registry entirely.
        _discard_session(session_id)
        raise ToolError(
            f"Download session '{session_id}' was cancelled and cannot be "
            f"reused. Create it again with download_create_session."
        )

    return manager


def _discard_session(session_id: str) -> dict | None:
    """Remove a session from the registry and release what it still holds.

    Called when a session is cancelled or closed, from whichever path noticed:
    the ``download_cancel`` tool, the progress panel's Cancel button, or
    ``download_close_session``. Removing the entry is what makes the identifier
    reusable, and ``purge`` drops the queue and the worker references the
    manager was still holding.

    Args:
        session_id: Session to forget.

    Returns:
        dict | None: The session's final status, or None when it was already
        gone.
    """
    with _sessions_lock:
        manager = _sessions.pop(session_id, None)

    if manager is None:
        return None

    status = manager.get_status()
    manager.purge()

    return status


def register_download_tools(server: MCPServer) -> None:
    """Register download queue tools.

    Args:
        server: Server instance the tools are attached to.
    """

    @server.tool(
        name="download_list_allowed_domains",
        title="List allowed download domains",
        description=(
            "List every host a download may come from. Call it before "
            "queueing a URL from an unfamiliar host: MSHCore validates the "
            "host itself and rejects anything else before queueing, so this "
            "only saves a failed call. No prerequisites. Read-only. Returns a "
            "flat list of strings — exact hostnames first, then wildcard "
            "entries written literally as '*.example.com', meaning that "
            "domain and any subdomain of it. A host not matching an entry on "
            "this list is rejected by download_file, download_add and "
            "download_add_many. Only http and https URLs are accepted at all. "
            "Short fixed-size response; no pagination."
        ),
        annotations=READ_ONLY,
    )
    @surface_core_errors
    def download_list_allowed_domains() -> list[str]:
        return allowed_sources()

    @server.tool(
        name="download_create_session",
        title="Create a download session",
        description=(
            "STEP 1 of 3 for downloading several files as one unit of work: "
            "create an empty named queue. Nothing is downloaded here — follow "
            "with download_add or download_add_many for every file, then "
            "download_start to begin "
            "transferring. For a single file use download_file instead, which "
            "does all three steps itself. session_id is a name you choose; it "
            "is the handle every other download_* tool takes, and it is NOT a "
            "progress_id, so never pass it to progress_get_status. Fails if "
            "that name is already in use — list the open ones with "
            "download_list_sessions. download_directory is where completed "
            "files land; it defaults to %LOCALAPPDATA%\\MSH\\downloads, is "
            "created with its parents if missing, and a relative path resolves "
            "against this server's working directory. max_retries is total "
            "attempts per file, not extra ones. Returns the new session's "
            "initial status, with an empty 'downloads' list."
        ),
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=False,
            open_world_hint=False,
        ),
    )
    @surface_core_errors
    def download_create_session(
        session_id: str,
        download_directory: str = DEFAULT_DOWNLOAD_DIRECTORY,
        max_retries: int = 3,
    ) -> dict:
        """Create a download session.

        Args:
            session_id: Identifier for later calls.
            download_directory: Destination directory for completed files.
                Defaults to %LOCALAPPDATA%\\MSH\\downloads.
            max_retries: Retry attempts per file before it is marked failed.

        Returns:
            dict: Initial session status.
        """
        with _sessions_lock:
            if session_id in _sessions:
                raise ToolError(
                    f"Download session already exists: '{session_id}'"
                )

            manager = DownloadManager(
                download_directory=download_directory,
                max_retries=max_retries,
            )
            _sessions[session_id] = manager

        return manager.get_status()

    @server.tool(
        name="download_list_sessions",
        title="List download sessions",
        description=(
            "List every open download session with its full status. Use it to "
            "recover a forgotten session_id, to check which name is free "
            "before download_create_session, or to see what is still "
            "transferring. No prerequisites. Read-only. Returns a dict keyed "
            "by session_id, each value being the same status dict "
            "download_get_status returns — including the per-file 'downloads' "
            "list, so the response grows with the total number of queued "
            "files across all sessions and there is no pagination; prefer "
            "download_get_status for one known session. Cancelled and closed "
            "sessions are absent: their identifiers are free for reuse. "
            "Sessions live in memory only and none survives a restart of this "
            "server, whereas a progress_id remains valid for "
            "progress_get_status across restarts."
        ),
        annotations=READ_ONLY,
    )
    @surface_core_errors
    def download_list_sessions() -> dict:
        with _sessions_lock:
            items = list(_sessions.items())

        return {
            session_id: manager.get_status()
            for session_id, manager in items
        }

    @server.tool(
        name="download_add",
        title="Queue a file",
        description=(
            "STEP 2 of 3: append one URL to an existing session's queue, with "
            "control over its filename. ENQUEUES ONLY — nothing is "
            "transferred and no network request is made until "
            "download_start is called on the "
            "same session. Requires download_create_session first: session_id "
            "is that session's name, never a progress_id. url must be http or "
            "https on a host from download_list_allowed_domains; a rejected "
            "host fails immediately and queues nothing. filename overrides "
            "the name taken from the URL; any directory part is stripped, so "
            "files always land directly in the session's download directory. "
            "An existing file is never overwritten — a taken name becomes a "
            "numbered variant, and the name actually reserved appears as "
            "'filename' in the returned status. Queue every file before "
            "starting; adding to a session that is already running is not how "
            "a queue is extended safely. Returns the session's status after "
            "queueing."
        ),
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=False,
            open_world_hint=False,
        ),
    )
    @surface_core_errors
    def download_add(
        session_id: str,
        url: str,
        filename: str | None = None,
    ) -> dict:
        """Queue one file for download.

        Args:
            session_id: Target session.
            url: HTTP or HTTPS URL on an allowed domain.
            filename: Optional destination filename.

        Returns:
            dict: Session status after queueing.
        """
        manager = _get_session(session_id)
        manager.add(url=url, filename=filename)
        return manager.get_status()

    @server.tool(
        name="download_add_many",
        title="Queue several files",
        description=(
            "STEP 2 of 3, batched: append several URLs to an existing "
            "session's queue in order, each named after its own URL. ENQUEUES "
            "ONLY — nothing is transferred until "
            "download_start is called. Requires download_create_session "
            "first: session_id is that session's name, never a progress_id. "
            "Use download_add instead when any file needs an explicit "
            "filename. NOT ATOMIC: validation happens per URL as the list is "
            "walked, so the first host outside download_list_allowed_domains "
            "fails the call with the earlier URLs left queued and the rest "
            "not queued at all — on failure, read the returned error and "
            "inspect download_get_status before retrying, or the survivors "
            "will be downloaded alongside a second attempt. Duplicate URLs "
            "are not detected and would be fetched twice under different "
            "names. Returns the session's status after queueing."
        ),
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=False,
            open_world_hint=False,
        ),
    )
    @surface_core_errors
    def download_add_many(session_id: str, urls: list[str]) -> dict:
        """Queue several files for download.

        Args:
            session_id: Target session.
            urls: URLs on allowed domains.

        Returns:
            dict: Session status after queueing.
        """
        manager = _get_session(session_id)
        manager.add_many(urls=urls)
        return manager.get_status()

    @server.tool(
        name="download_get_status",
        title="Get download progress",
        description=(
            "Report one download session's state and every queued file's "
            "progress. This is the queue's own view, keyed by session_id — "
            "the counterpart for a progress_id is progress_get_status, and "
            "the two identifiers are not interchangeable. Requires the "
            "session to still be open. Read-only, fast, and safe to poll "
            "repeatedly. Returns one dict with 'running', 'paused', "
            "'cancelled', 'closed', 'files_deleted', 'cancel_reason', "
            "'current_index' (0-based, -1 before the first file starts), "
            "'total_files' and 'downloads'. Each 'downloads' entry has 'url', "
            "'filename' (the name actually being written, which may differ "
            "from the one requested), 'requested_filename' (set only when "
            "renamed), 'status', 'downloaded' and 'total' in bytes with "
            "'total' null when the server sent no length, 'speed' in bytes "
            "per second as a running average, and 'error'. A file's 'status' "
            "is one of 'waiting', 'connecting', 'downloading', 'paused', "
            "'retrying', 'completed', 'skipped', 'failed' or 'cancelled'. The "
            "response grows with the queue length and has no pagination, so "
            "keep queues to a reasonable size. 'files_deleted' true means a "
            "cancellation intends to delete this session's files, not that "
            "deletion has finished."
        ),
        annotations=READ_ONLY,
    )
    @surface_core_errors
    def download_get_status(session_id: str) -> dict:
        """Get a session's current status.

        Args:
            session_id: Target session.

        Returns:
            dict: Manager state and per-file progress.
        """
        return _get_session(session_id).get_status()

    @server.tool(
        name="download_pause",
        title="Pause downloading",
        description=(
            "Suspend a session's active transfer without cancelling it. "
            "Non-destructive: the queue, the files already completed and the "
            "active file's partial data are all kept, so download_resume "
            "continues from where it stopped rather than restarting. Requires "
            "a session that download_start "
            "has started; session_id is the queue's name, never a progress_id "
            "— the progress_id equivalent is progress_pause. Returns "
            "immediately; the transfer stops at its next chunk boundary, so "
            "the returned status may still show 'downloading' for a moment. "
            "Does nothing when the session is not running or is already "
            "paused. To stop a session and discard what it produced, use "
            "download_cancel instead."
        ),
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        ),
    )
    @surface_core_errors
    def download_pause(session_id: str) -> dict:
        """Pause a session's active download.

        Args:
            session_id: Target session.

        Returns:
            dict: Session status after pausing.
        """
        manager = _get_session(session_id)
        manager.pause()
        return manager.get_status()

    @server.tool(
        name="download_resume",
        title="Resume downloading",
        description=(
            "Continue a session paused by download_pause. Requires that "
            "session to be paused; session_id is the queue's name, never a "
            "progress_id. The active file continues from its partial data via "
            "an HTTP range request rather than starting over, so no bytes are "
            "refetched and nothing already on disk is discarded. Returns "
            "immediately with the session's status; the transfer picks up on "
            "its background thread. Does nothing when the session is not "
            "paused, so it cannot be used to start a queue that was never "
            "started — use download_start for "
            "that. It cannot revive a cancelled session either: that requires "
            "creating the session again."
        ),
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=True,
        ),
    )
    @surface_core_errors
    def download_resume(session_id: str) -> dict:
        """Resume a paused session.

        Args:
            session_id: Target session.

        Returns:
            dict: Session status after resuming.
        """
        manager = _get_session(session_id)
        manager.resume()
        return manager.get_status()

    @server.tool(
        name="download_skip",
        title="Skip the current file",
        description=(
            "Abandon the file currently transferring and move on to the next "
            "in the queue. Use it for one file that is stalled or unwanted; "
            "the rest of the queue continues, unlike download_cancel which "
            "abandons everything. Requires a running session; session_id is "
            "the queue's name, never a progress_id. The skipped file is "
            "marked 'skipped' and is not retried by a later download_start, "
            "and its partial data is left on disk rather than deleted — clean "
            "it up yourself if it matters, or use download_cancel to have the "
            "session's files removed. Returns immediately with the session's "
            "status; the skip takes effect at the next chunk boundary. "
            "Calling it between files, when nothing is transferring, applies "
            "the skip to the next file instead — that file is fetched in "
            "full first and only then marked skipped."
        ),
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=True,
            idempotent_hint=False,
            open_world_hint=False,
        ),
    )
    @surface_core_errors
    def download_skip(session_id: str) -> dict:
        """Skip a session's current file.

        Args:
            session_id: Target session.

        Returns:
            dict: Session status after requesting the skip.
        """
        manager = _get_session(session_id)
        manager.skip()
        return manager.get_status()

    @server.tool(
        name="download_cancel",
        title="Cancel all downloads",
        description=(
            "Stop a session's active transfer, abandon the rest of its queue, "
            "and delete the files it produced. IRREVERSIBLE AND DELETES "
            "COMPLETED FILES: both partial data and files this session had "
            "already finished are removed, because the queue is one unit of "
            "work that did not complete. Files that existed before the "
            "session started are never touched, and a queue that had already "
            "finished everything loses nothing. Pass keep_files=true to "
            "abandon the queue while leaving every downloaded file in place. "
            "Requires an open session; session_id is the queue's name, never "
            "a progress_id — the progress_id equivalent is progress_cancel. "
            "To suspend instead of destroying, use download_pause. The "
            "cancellation is recorded in the execution log, where logs_read "
            "will show it. The session is then removed: its identifier "
            "becomes free, it refuses all further use, and downloading the "
            "same files again means calling download_create_session and "
            "download_add from scratch. Blocks up to 60 seconds so the "
            "returned status reflects the completed cleanup."
        ),
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=True,
            idempotent_hint=True,
            open_world_hint=False,
        ),
    )
    @surface_core_errors
    def download_cancel(
        session_id: str,
        keep_files: bool = False,
    ) -> dict:
        """Cancel a session's remaining downloads and remove the session.

        Args:
            session_id: Target session.
            keep_files: Keep what the session downloaded instead of deleting it.

        Returns:
            dict: Session status after the cancellation and cleanup.
        """
        manager = _get_session(session_id)
        # MSHCore is told first, because its first cancellation is the one that
        # decides whether the files go: notifying the panel's job first would
        # cancel through the job's own canceller, which always deletes.
        manager.cancel(cleanup=not keep_files)
        # A progress bar over this session would otherwise keep offering Cancel
        # and Stop for a queue that is being torn down.
        note_download_ended(
            session_id,
            reason=(
                "Cancelled with download_cancel; downloaded files kept."
                if keep_files
                else "Cancelled with download_cancel; downloaded files removed."
            ),
        )
        # The worker stops at a chunk boundary, so the status is only final once
        # it has actually exited and MSHCore's cleanup has run.
        manager.wait_until_stopped(timeout=CANCEL_WAIT_SECONDS)

        status = _discard_session(session_id) or manager.get_status()

        return status

    @server.tool(
        name="download_close_session",
        title="Close a download session",
        description=(
            "Finish with a download session and drop it, keeping every file "
            "it downloaded. This is the tidy end to a completed session and "
            "the safe way to stop one without losing data; download_cancel "
            "deletes the session's files instead. Requires an open session; "
            "session_id is the queue's name, never a progress_id. A "
            "still-running transfer is stopped first, and both completed and "
            "partial files are left on disk. What is discarded is only "
            "in-memory state: the queue and its progress history, which are "
            "unrecoverable, so read download_get_status first if that record "
            "is wanted. The session identifier becomes free for reuse "
            "afterwards and the session itself refuses all further calls. "
            "Blocks up to 60 seconds waiting for the transfer to stop, then "
            "returns the session's final status."
        ),
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=True,
            idempotent_hint=True,
            open_world_hint=False,
        ),
    )
    @surface_core_errors
    def download_close_session(session_id: str) -> dict:
        """Close and forget a download session.

        Args:
            session_id: Target session.

        Returns:
            dict: Final status of the closed session.
        """
        manager = _get_session(session_id)
        # Closing a session is bookkeeping, not a request to undo the download,
        # so the files it produced are left alone. MSHCore is told before the panel
        # for the same reason as in download_cancel: the job's own canceller
        # deletes, and the first cancellation is the one that decides.
        manager.close()
        note_download_ended(
            session_id,
            reason="Session closed; downloaded files kept.",
        )
        manager.wait_until_stopped(timeout=CANCEL_WAIT_SECONDS)

        return _discard_session(session_id) or manager.get_status()


# ============================================================
# Logging — MSHCore.logging
# ============================================================

# MSHCore.logging.write_log is deliberately not exposed: MSHCore writes its own
# entries, and letting a client inject records would pollute the history.

def register_logging_tools(server: MCPServer) -> None:
    """Register execution log tools.

    Args:
        server: Server instance the tools are attached to.
    """

    @server.tool(
        name="logs_read",
        title="Read the execution log",
        description=(
            "Read entries from this project's execution log, which MSHCore "
            "writes for every significant operation. PRIMARY diagnostic tool "
            "after any failure: a tool's error message is often shorter than "
            "the log entry behind it. This is MSHCore's own log; "
            "ollama_read_log serves Ollama's separate log files, which are "
            "where an Ollama service, model-load or GPU-detection failure is "
            "explained. No prerequisites, though logs_get_file_info first "
            "tells you how large the log is. Read-only. Returns a list of "
            "entries, each with 'timestamp', 'level', 'component', 'action', "
            "'message' and a 'details' object. The level, component and "
            "action filters are exact-match, case-sensitive and combine as "
            "AND, so level='ERROR' with component='download_manager' returns "
            "only failed downloads; 'ERROR' matches and 'error' does not. "
            "PAGINATE with line_count, which caps how many matching entries "
            "come back and keeps the newest ones, since the log is appended "
            "chronologically — pass it whenever the log is large, because an "
            "uncapped read returns every match and will overflow the context. "
            "Entries are always returned oldest first, capped or not. "
            "line_count must be 1 or greater. Returns an empty list when "
            "nothing matches or the log does not exist yet."
        ),
        annotations=READ_ONLY,
    )
    @surface_core_errors
    def logs_read(
        level: str | None = None,
        component: str | None = None,
        action: str | None = None,
        line_count: int | None = None,
    ) -> list[dict]:
        """Read filtered execution log entries.

        Args:
            level: Optional severity, for example 'INFO', 'WARNING', 'ERROR'.
            component: Optional component, for example 'ollama/runtime',
                'ollama/model', 'system/scanner', 'python', 'download_manager'.
            action: Optional action, for example 'start', 'run', 'download_failed'.
            line_count: Optional maximum number of entries, counted back from
                the newest match.

        Returns:
            list[dict]: Matching log entries in file order, oldest first.
        """
        return core_logging.read_logs(
            level=level,
            component=component,
            action=action,
            line_count=line_count,
        )

    @server.tool(
        name="logs_get_file_info",
        title="Get the execution log's path and size",
        description=(
            "Describe the execution log without reading any of it. Call it "
            "before logs_read to size the request: the entry count is what "
            "tells you whether an uncapped read is safe or line_count is "
            "required to avoid overflowing the context. Read-only in effect, "
            "though it does create the log's directory if absent. Returns one "
            "dict with exactly 'path' (the log's absolute path, which is "
            "%LOCALAPPDATA%\\MSH\\logs\\executions.log, for reading or "
            "archiving the raw file outside these tools), "
            "'line_count' (physical lines, which can slightly exceed the "
            "number of entries logs_read yields because unparseable lines are "
            "counted here and skipped there) and 'size_bytes'. A log that has "
            "never been written reports zero for both counts rather than "
            "failing. Fixed-size response; no pagination."
        ),
        annotations=READ_ONLY,
    )
    @surface_core_errors
    def logs_get_file_info() -> dict:
        info = core_logging.get_log_file_info()

        return {**info, "path": str(info["path"])}


# ============================================================
# Server
# ============================================================

REGISTRARS = (
    register_system_tools,
    register_ollama_runtime_tools,
    register_ollama_model_tools,
    register_benchmark_tools,
    register_python_tools,
    register_download_tools,
    register_logging_tools,
    # Reading and controlling a progress bar. Registered here rather than on the
    # Apps extension because a tool bound to the panel is drawn every time the
    # model calls it, and the model polls these repeatedly by design.
    register_progress_tools,
)


def create_server() -> MCPServer:
    """Build the MCP server with every tool registered.

    Returns:
        MCPServer: Configured server instance, ready to run.
    """
    server = MCPServer(
        name=SERVER_NAME,
        title=SERVER_TITLE,
        instructions=INSTRUCTIONS,
        version=SERVER_VERSION,
        # The progress panel is an additive MCP Apps extension: it contributes the
        # ui:// resource and the tools bound to it, and intercepts nothing. The
        # session registry's three operations are handed over so the starting
        # tools bound to the panel — download_file and download_start — act on
        # the same sessions the plain download tools do.
        extensions=[
            create_progress_app(
                get_session=_get_session,
                release_session=_discard_session,
                register_session=_register_session,
            )
        ],
    )

    for register in REGISTRARS:
        register(server)

    return server


def main() -> None:
    """Build the server and serve it over stdio."""
    create_server().run(transport="stdio")


if __name__ == "__main__":
    main()
