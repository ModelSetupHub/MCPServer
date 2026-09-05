# ModelSetupHub — MCPServer

**An MCP server that hands the whole ModelSetupHub toolkit to an AI agent.**

English · [فارسی](README.fa.md)

MCPServer lets an AI assistant set up local AI models for you. Instead of choosing a model, working out whether it fits
your GPU, and hunting for the parameters that make it fast, you describe the goal — *install a coding model that suits my
system* — and the agent uses these tools to inspect your hardware, manage Ollama, download and configure models, and
benchmark settings against each other.

It speaks the Model Context Protocol, so it plugs into any MCP-capable client, Claude Desktop included.

## Where this fits

ModelSetupHub is three repositories over one engine.

| Repository | What it is | Reach for it when |
| --- | --- | --- |
| [Core](https://github.com/ModelSetupHub/Core) | The Python library that does the work | You are writing your own scripts or automation |
| [WebApp](https://github.com/ModelSetupHub/WebApp) | A local web dashboard over the same functions | You want a graphical interface and full manual control |
| **MCPServer** (this one) | An MCP server that hands the same tools to an AI agent | You would rather describe the goal in plain language |

This is the friendlier way in: the agent decides the order of operations and explains what it is doing. The dashboard is
the counterpart for people who already know exactly which model and which parameters they want. Both drive the identical
library, so they can be used interchangeably on the same machine.

## What the agent can do

56 tools in total. All of them are direct pass-throughs to the toolkit, so the agent gets the same capabilities a
script would, grouped by area:

- **System** — scan the machine, or read just the memory, storage, GPU or CUDA details.
- **Ollama runtime** — check whether Ollama is installed and running, start and stop it, install Ollama itself, and read
  Ollama's own log files when something failed and the reason is buried in them.
- **Models** — list what is installed and what is loaded, inspect a model, run a prompt, preload and unload, register a
  local GGUF file, create a configured copy of a model, and delete one.
- **Benchmarking** — run a set of prompts under one configuration, or compare several configurations or several models
  over the same prompts, with per-prompt timing-to-first-token, VRAM and GPU readings, repetition averaging with
  run-to-run noise stats, a significance verdict on every comparison, and a saved history of past runs to browse, re-read
  or delete.
- **Python** — create and remove virtual environments, install, uninstall and list packages, write, read, edit, delete
  and run scripts, and install Python itself from an official installer.
- **Downloads** — fetch a single file in one call, or build a queue, start it, watch its progress, pause, resume, skip a
  file or cancel the lot. Downloads are limited to a fixed list of trusted hosts, and there is a tool that reports which
  ones. An existing file is never overwritten: a colliding name is saved as `model-1.safetensors` rather than silently
  reported as already downloaded.
- **Logs** — read back the toolkit's execution log, filtered, or just check its size before reading it.

Every tool carries annotations so a client can tell a read-only call from one that changes something on your machine.
Writing to the execution log is deliberately not exposed — the toolkit writes its own entries, and letting a client inject
records would corrupt the history.

## The progress panel

Downloads and benchmarks take minutes, which is too long for a chat to sit silent. Those operations have a variant that
draws a **live progress bar inside the conversation itself**, next to the assistant's message: an overall percentage,
one row per file or per configuration, and the current transfer rate.

The panel carries its own controls, and they do different things:

- **Cancel** ends the operation *and undoes it*. A cancelled download deletes the files that download produced; a
  cancelled benchmark discards its partial measurements and unloads the model it had loaded. It cannot be undone.
- **Stop** appears for downloads only and merely suspends the transfer. The queue and the data already fetched are kept,
  and pressing it again continues the current file from where it left off.

The bar keeps working across a page reload or a reopened conversation, because its state is stored on disk rather than
held in memory. Operations with nothing measurable to report — installations, a hardware scan — have no progress variant.

## Requirements

- Python 3.10 or newer
- Windows for the full feature set. Hardware detection relies on PowerShell and WMI, and GPU detection on `nvidia-smi`.
- Git, to fetch the `Core` submodule
- An MCP-capable client

Dependencies are listed in `requirements.txt`. The toolkit itself comes from the `Core` submodule and brings its own.

## Setup

The toolkit lives in the `Core` submodule and has to be installed as a package first. A bundled helper does it for you:

```bash
git clone --recurse-submodules https://github.com/ModelSetupHub/MCPServer.git
cd MCPServer

# already cloned without --recurse-submodules? fetch it now:
git submodule update --init

python utils/install_mshcore.py
```

Or install it directly with `pip install ./Core`. Then the server's own dependencies:

```bash
pip install -r requirements.txt
```

The helper also takes `--check` to report whether the toolkit is already installed, `--editable` for a development
install, and `--force` to reinstall — useful after pulling a newer submodule commit.

## Running it

```bash
python main.py
```

The server communicates over standard input and output, so a client starts it for you. Register it in your client's
configuration:

```json
{
  "mcpServers": {
    "modelsetuphub": {
      "command": "python",
      "args": ["C:/path/to/MCPServer/main.py"]
    }
  }
}
```

If Claude Desktop is your client, `utils/claude_setup.py` does that registration for you. It is a small standalone
desktop app — not an MCP tool — that can install Claude Desktop, find your Python interpreter and configuration file
automatically, and add or remove MCP servers, backing the configuration up before every change. It needs Windows for
detection and installation:

```bash
python utils/claude_setup.py
```

## Good to know

- **These tools change your machine.** They install software, run installers, delete models and remove directories.
  Reviewing what the agent proposes before approving it is worth the moment it costs.
- **Deleting a model is not recoverable**, and cancelling a download removes the files that download produced.
- **A cancelled download session cannot be restarted.** Downloading the same files again means starting a new session,
  which is what stops the same queue being run twice.
- **Errors come through verbatim.** When the toolkit refuses something, the real reason reaches the client rather than a
  generic failure — a blocked download domain says so by name.

## Status and licence

Early but working, and under active development. Issues and pull requests are welcome — behaviour changes usually belong
in [Core](https://github.com/ModelSetupHub/Core), since this server mostly forwards to it.

No licence file has been added yet. If you need one before using this in your own project, open an issue.
