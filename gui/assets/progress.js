/* ModelSetupHub progress panel.
 *
 * An MCP Apps view: it speaks the ext-apps postMessage JSON-RPC dialect against
 * window.parent, so it needs no bundler and no network access. progress.css and
 * progress.js are inlined into progress.html by gui/loader.py.
 *
 * The panel is deliberately dumb. It learns one progress_id from its tool result,
 * polls with it, and draws whatever comes back. It derives no identifiers,
 * reconstructs no state, and decides nothing about lifecycle: only a snapshot whose
 * status is completed, failed or cancelled stops the polling. A timeout, a missing
 * snapshot or a slow server means retry, never "ended".
 *
 * The tools below are the panel's own, app-visible ones — not the tools the model
 * calls. They are separate names for the same server functions, so this view can
 * poll twice a second without any of it reaching the conversation.
 */

(function () {
  "use strict";

  var PROTOCOL_VERSION = "2026-01-26";
  var APP_INFO = { name: "modelsetuphub-progress", version: "0.2.0" };

  var STATUS_TOOL = "progress_panel_status";
  var CANCEL_TOOL = "progress_panel_cancel";
  var PAUSE_TOOL = "progress_panel_pause";

  var POLL_INTERVAL_MS = 700;
  // A poll that never comes back must not wedge the poller, so each one is
  // abandoned after this and the next tick tries again. It also bounds how stale
  // the panel can be once a stalled transport recovers.
  var POLL_TIMEOUT_MS = 5000;

  // How long a view waits for a starting tool's result before concluding that no
  // operation is coming. A live one arrives within milliseconds of the panel
  // opening — the id exists before the tool returns — so anything past a few
  // seconds is a view with nothing behind it: a conversation reopened without its
  // result, or a panel a host drew for something that is not a starting call.
  // Without this it would animate "Starting…" for ever and read as work in
  // progress.
  var ADOPT_GRACE_MS = 6000;

  var TERMINAL = { completed: 1, failed: 1, cancelled: 1 };

  var pending = {};
  var nextId = 1;
  var hostCapabilities = {};

  // The one thing this panel knows about its operation.
  var progressId = null;

  var pollTimer = null;
  var adoptTimer = null;
  var inFlight = false;
  var stopped = false;
  var busy = false;

  var dom = {};

  /* ---------------------------------------------------------------- transport */

  function send(message) {
    window.parent.postMessage(message, "*");
  }

  function request(method, params, timeout) {
    var id = nextId++;

    return new Promise(function (resolve, reject) {
      var timer =
        timeout &&
        window.setTimeout(function () {
          delete pending[id];
          reject(new Error("Timed out waiting for " + method));
        }, timeout);

      pending[id] = {
        settle: function (fn, value) {
          if (timer) {
            window.clearTimeout(timer);
          }
          fn(value);
        },
        resolve: resolve,
        reject: reject,
      };

      send({ jsonrpc: "2.0", id: id, method: method, params: params || {} });
    });
  }

  function notify(method, params) {
    send({ jsonrpc: "2.0", method: method, params: params || {} });
  }

  window.addEventListener("message", function (event) {
    var message = event.data;

    if (!message || message.jsonrpc !== "2.0") {
      return;
    }

    // A reply to something this panel sent.
    if (message.id !== undefined && message.method === undefined) {
      var waiter = pending[message.id];
      delete pending[message.id];

      if (!waiter) {
        return;
      }

      if (message.error) {
        waiter.settle(waiter.reject, new Error(message.error.message || "Host error"));
      } else {
        waiter.settle(waiter.resolve, message.result);
      }
      return;
    }

    if (message.method === undefined) {
      return;
    }

    // A request from the host must be answered. ui/ping and ui/resource-teardown
    // are the ones a passive view sees.
    if (message.id !== undefined) {
      send({ jsonrpc: "2.0", id: message.id, result: {} });

      if (message.method === "ui/resource-teardown") {
        stopPolling();
        cancelAdoptTimer();
      }
      return;
    }

    handle(message.method, message.params || {});
  });

  function handle(method, params) {
    if (method === "ui/notifications/tool-result") {
      adopt(params);
      return;
    }

    if (method === "ui/notifications/host-context-changed") {
      applyTheme(params);
    }
  }

  function applyTheme(context) {
    if (!context) {
      return;
    }

    if (context.theme === "light" || context.theme === "dark") {
      document.documentElement.setAttribute("data-theme", context.theme);
    }

    var variables = context.styles && context.styles.variables;

    if (variables) {
      Object.keys(variables).forEach(function (name) {
        var value = variables[name];
        if (typeof value === "string" && name.indexOf("--") === 0) {
          document.documentElement.style.setProperty(name, value);
        }
      });
    }
  }

  // The tool result carries this panel's progress_id. The host replays it when a
  // conversation is reopened, so this is also how a restored panel finds its
  // operation — the same id, the same polling, no separate restoration path.
  //
  // Only a starting tool's result carries `progress_id`; a status snapshot carries
  // `id` instead, and is ignored here. And the first id adopted is the panel's for
  // good: this view belongs to one operation, so a later result naming a different
  // one is not this bar's business.
  function adopt(result) {
    var payload = payloadOf(result);
    var id = payload && payload.progress_id;

    if (typeof id !== "string" || !id || progressId !== null) {
      return;
    }

    cancelAdoptTimer();

    progressId = id;
    stopped = false;
    startPolling();
  }

  /* ----------------------------------------------------------------- adoption */

  // A view with no operation behind it. It has no id, so there is nothing to poll
  // and nothing will ever arrive; saying so is the only honest frame. Left on
  // "Starting…" it reads as a task that is still running, which is what several
  // idle bars beside one real one looked like.
  function startAdoptTimer() {
    if (adoptTimer !== null || progressId !== null) {
      return;
    }

    adoptTimer = window.setTimeout(function () {
      adoptTimer = null;

      if (progressId === null) {
        renderIdle();
      }
    }, ADOPT_GRACE_MS);
  }

  function cancelAdoptTimer() {
    if (adoptTimer !== null) {
      window.clearTimeout(adoptTimer);
      adoptTimer = null;
    }
  }

  function renderIdle() {
    stopped = true;
    stopPolling();

    dom.title.textContent = "No operation to show";
    setText(
      dom.message,
      "This panel has no progress to follow. Progress is reported by the call " +
        "that starts a download, benchmark or model import."
    );

    dom.badge.dataset.state = "completed";
    dom.badge.textContent = "idle";

    dom.bar.dataset.state = "completed";
    dom.bar.dataset.indeterminate = "false";
    dom.barFill.style.width = "0%";

    dom.cancel.classList.add("hidden");
    dom.pause.classList.add("hidden");
    dom.metrics.classList.add("hidden");
    dom.steps.classList.add("hidden");
  }

  /* ------------------------------------------------------------------ polling */

  function startPolling() {
    if (pollTimer !== null || progressId === null || !hostCapabilities.serverTools) {
      return;
    }

    poll();
    pollTimer = window.setInterval(poll, POLL_INTERVAL_MS);
  }

  function stopPolling() {
    if (pollTimer !== null) {
      window.clearInterval(pollTimer);
      pollTimer = null;
    }
  }

  function poll() {
    if (inFlight || stopped || progressId === null) {
      return;
    }

    inFlight = true;

    request(
      "tools/call",
      { name: STATUS_TOOL, arguments: { progress_id: progressId } },
      POLL_TIMEOUT_MS
    )
      .then(function (result) {
        var snapshot = payloadOf(result);

        // Nothing usable: a transport hiccup, or a record not written yet. Keep
        // whatever is on screen and try again — this is never an ending.
        if (snapshot && snapshot.found !== false) {
          render(snapshot);
        }
      })
      .catch(function () {
        // Same rule for a timeout or a host error. The operation is unaffected by
        // this panel failing to read it.
      })
      .then(function () {
        inFlight = false;
      });
  }

  function payloadOf(result) {
    if (!result) {
      return null;
    }

    var structured = result.structuredContent;

    if (structured && typeof structured === "object") {
      // Some hosts wrap a tool's object return under `result`. Both shapes carry
      // the same snapshot, so both are accepted rather than one being silently
      // unreadable.
      return structured.result !== undefined && structured.id === undefined
        ? structured.result
        : structured;
    }

    var blocks = result.content || [];

    for (var index = 0; index < blocks.length; index += 1) {
      if (blocks[index] && blocks[index].type === "text") {
        try {
          return JSON.parse(blocks[index].text);
        } catch (error) {
          return null;
        }
      }
    }

    return null;
  }

  /* ------------------------------------------------------------------ controls */

  function control(tool) {
    if (busy || progressId === null) {
      return;
    }

    busy = true;
    dom.cancel.disabled = true;
    dom.pause.disabled = true;

    request("tools/call", {
      name: tool,
      arguments: { progress_id: progressId },
    })
      .then(function (result) {
        var snapshot = payloadOf(result);

        if (snapshot && snapshot.found !== false) {
          render(snapshot);
        }
      })
      .catch(function (error) {
        setText(dom.message, error.message);
      })
      .then(function () {
        busy = false;
        // The next snapshot decides whether the buttons come back.
        poll();
      });
  }

  /* ----------------------------------------------------------------- rendering */

  // One renderer, one input: a snapshot from the server. Nothing here knows or
  // cares whether the operation is live or was read back from a record.
  function render(snapshot) {
    var status = snapshot.status || "running";

    // An import reports nothing while it runs: no bar, no steps, no status
    // text — the title and badge are the whole view. Only a failure earns
    // a line.
    var silent = snapshot.type === "importmodel";

    // Suspended and cancelling are flags on a running job, shown as their own
    // badge because that is what the user is looking for.
    var shown = status;

    if (status === "running" || status === "starting") {
      if (snapshot.cancelling) {
        shown = "cancelling";
      } else if (snapshot.paused) {
        shown = "paused";
      }
    }

    var percent = snapshot.progress;
    var determinate = percent !== null && percent !== undefined;

    dom.title.textContent = snapshot.title || titleFor(snapshot);

    if (silent && !TERMINAL[status]) {
      setText(dom.message, "");
    } else {
      setText(dom.message, snapshot.error || snapshot.message);
    }

    dom.badge.dataset.state = shown;
    dom.badge.textContent = shown;

    dom.bar.classList.toggle("hidden", silent);
    dom.bar.dataset.state = shown;
    dom.bar.dataset.indeterminate =
      !silent && !determinate && !TERMINAL[status] ? "true" : "false";
    dom.barFill.style.width = clamp(percent) + "%";

    renderControls(snapshot, status, silent);
    renderMetrics(snapshot, silent);
    renderSteps(silent ? [] : snapshot.steps || []);

    // The only thing that ends the polling.
    if (TERMINAL[status]) {
      stopped = true;
      stopPolling();
    }
  }

  function titleFor(snapshot) {
    if (snapshot.type === "download") {
      return "Downloading";
    }

    if (snapshot.type === "importmodel") {
      return "Importing model";
    }

    return "Benchmarking";
  }

  function renderControls(snapshot, status, activity) {
    // An import offers no Cancel: it is one disk-bound copy whose
    // interruption buys nothing, so its view carries no buttons at all.
    var canCancel = snapshot.can_cancel === true && !busy && !activity;
    var canPause = snapshot.can_pause === true && !busy && !activity;

    dom.cancel.classList.toggle("hidden", !canCancel);
    dom.pause.classList.toggle("hidden", !canPause);

    if (canCancel) {
      dom.cancel.disabled = false;
      dom.cancel.textContent = "Cancel";
      dom.cancel.title = "Cancel this task and undo what it has done";
    }

    if (canPause) {
      dom.pause.disabled = false;
      dom.pause.textContent = snapshot.paused ? "Resume" : "Stop";
      dom.pause.title = snapshot.paused
        ? "Continue the download from where it stopped"
        : "Stop the download without cancelling the task";
    }

    if (TERMINAL[status]) {
      dom.cancel.classList.add("hidden");
      dom.pause.classList.add("hidden");
    }
  }

  function renderMetrics(snapshot, silent) {
    var parts = [];
    var percent = snapshot.progress;

    // A silent view has no meaningful percentage: nothing under the title
    // would read as progress.
    if (!silent && percent !== null && percent !== undefined) {
      parts.push(pair(Math.round(clamp(percent)) + "%", "complete"));
    }

    (snapshot.metrics || []).forEach(function (metric) {
      parts.push(pair(metric.value, metric.label));
    });

    dom.metrics.replaceChildren.apply(dom.metrics, parts);
    dom.metrics.classList.toggle("hidden", parts.length === 0);
  }

  function pair(value, label) {
    var wrapper = document.createElement("span");
    var strong = document.createElement("span");

    strong.className = "value";
    strong.textContent = String(value === undefined || value === null ? "—" : value);
    wrapper.appendChild(strong);

    if (label) {
      wrapper.appendChild(document.createTextNode(" " + label));
    }

    return wrapper;
  }

  function renderSteps(steps) {
    var items = steps.map(function (step) {
      var item = document.createElement("li");
      item.className = "step";
      item.dataset.state = step.state || "waiting";

      var name = document.createElement("span");
      name.className = "step-name";
      name.textContent = step.name || "step";
      item.appendChild(name);

      var value = document.createElement("span");
      value.className = "step-value";
      value.textContent = step.detail || step.state || "waiting";
      item.appendChild(value);

      var determinate = step.percent !== null && step.percent !== undefined;

      if (determinate || step.state === "running") {
        var bar = document.createElement("div");
        bar.className = "bar step-bar";
        bar.dataset.state = step.state || "waiting";
        bar.dataset.indeterminate =
          step.state === "running" && !determinate ? "true" : "false";

        var fill = document.createElement("div");
        fill.className = "bar-fill";
        fill.style.width = clamp(step.percent) + "%";
        bar.appendChild(fill);
        item.appendChild(bar);
      }

      if (step.error) {
        var note = document.createElement("span");
        note.className = "step-note";
        note.dataset.level = "error";
        note.textContent = step.error;
        item.appendChild(note);
      }

      return item;
    });

    dom.steps.replaceChildren.apply(dom.steps, items);
    dom.steps.classList.toggle("hidden", items.length === 0);
  }

  /* ---------------------------------------------------------------- utilities */

  function setText(element, value) {
    element.textContent = value || "";
    element.classList.toggle("hidden", !value);
  }

  function clamp(percent) {
    if (percent === null || percent === undefined || isNaN(percent)) {
      return 0;
    }

    return Math.max(0, Math.min(100, Number(percent)));
  }

  function reportSize() {
    notify("ui/notifications/size-changed", {
      height: Math.ceil(document.documentElement.getBoundingClientRect().height),
    });
  }

  /* ----------------------------------------------------------------- lifecycle */

  function main() {
    dom = {
      title: document.getElementById("title"),
      message: document.getElementById("subtitle"),
      badge: document.getElementById("badge"),
      cancel: document.getElementById("cancel"),
      pause: document.getElementById("pause"),
      bar: document.getElementById("bar"),
      barFill: document.getElementById("bar-fill"),
      metrics: document.getElementById("metrics"),
      steps: document.getElementById("steps"),
    };

    dom.cancel.addEventListener("click", function () {
      control(CANCEL_TOOL);
    });
    dom.pause.addEventListener("click", function () {
      control(PAUSE_TOOL);
    });

    request("ui/initialize", {
      appInfo: APP_INFO,
      appCapabilities: {},
      protocolVersion: PROTOCOL_VERSION,
    })
      .then(function (result) {
        hostCapabilities = (result && result.hostCapabilities) || {};
        applyTheme((result && result.hostContext) || {});
        notify("ui/notifications/initialized", {});

        if (!hostCapabilities.serverTools) {
          dom.badge.dataset.state = "failed";
          dom.badge.textContent = "unavailable";
          setText(
            dom.message,
            "This client does not allow the panel to read progress, so the " +
              "result will arrive in the chat instead."
          );
          return;
        }

        // The tool result may already have arrived; either way this is a no-op
        // until an id is known. The timer covers the case where it never does.
        startPolling();
        startAdoptTimer();
      })
      .catch(function (error) {
        setText(dom.message, error.message);
      });

    if (typeof ResizeObserver === "function") {
      new ResizeObserver(reportSize).observe(document.documentElement);
    }

    window.addEventListener("beforeunload", function () {
      stopPolling();
      cancelAdoptTimer();
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", main, { once: true });
  } else {
    main();
  }
})();
