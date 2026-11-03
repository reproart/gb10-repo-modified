"""Engine-side Prometheus sidecar for this fork's gauges (README patch 11).

vLLM serves /metrics from the API-server process, but the fork's interesting
counters live in the EngineCore process: the PLE mmap gather, the mamba
state-copy guard tripwire and the never-evict pin. Rather than plumbing them
through the SchedulerStats IPC channel (which is serialized and drops unknown
fields), this module serves them from the engine process itself with
prometheus_client — the same library vLLM uses — on a separate port:

  VLLM_CUSTOM_METRICS_PORT     inside-container port, default 18400; 0 = off.
                               scripts/serve-intel-ar.sh publishes it as
                               METRICS_PORT (host side, 0 = not published).
  VLLM_CUSTOM_METRICS_INTERVAL refresher period in seconds, default 5.

Metrics (all created up front, so they are visible as 0 even before traffic):
  vllm:ple_mmap_ops_total            lookup ops (counter)
  vllm:ple_mmap_op_ms_total          op wall time; rate()/rate(ops) = ms/op
  vllm:ple_mmap_gather_ms_total      disk-read share of the above
  vllm:ple_mmap_rows_total / vllm:ple_mmap_bytes_total
  vllm:ple_mmap_prefetch_hit_total / vllm:ple_mmap_prefetch_miss_total
  vllm:mamba_state_copy_guard_total  counter; tripwire, expect a constant 0
  vllm:never_evict_blocks_reserved / vllm:never_evict_pin_queue_blocks
  vllm:never_evict_pin_bytes
  vllm:torch_cuda_allocated_bytes    live tensors on the GPU (torch's caching allocator)
  vllm:torch_cuda_reserved_bytes     what that allocator holds: live tensors, CUDA graph
                                     pools and freed blocks it keeps for reuse; on GB10
                                     host memory, not given back without empty_cache()
  vllm:torch_cuda_max_reserved_bytes the peak of the above since the engine started
  vllm:torch_idle_released_bytes_total what the idle release below gave back

Idle release (VLLM_IDLE_EMPTY_CACHE_SEC, default 0 = off): once the scheduler has
had no request for that many seconds and the allocator holds at least
VLLM_IDLE_EMPTY_CACHE_MIN_MB (512) of freed blocks, the refresher thread calls
torch.cuda.empty_cache(): on GB10 those blocks are host memory, kept at the
peak of the last large request. CUDA graph pools stay (empty_cache leaves them).
It runs again only once later work has raised the allocator's hold by that much
again, so blocks it cannot free do not make it repeat. Each release logs a line.
It runs with VLLM_CUSTOM_METRICS_PORT=0 too (the thread then serves nothing).
Once per idle period it also calls glibc's malloc_trim(0) (VLLM_IDLE_MALLOC_TRIM,
default 1): the engine's freed heap (model loading, Python objects) goes back to
the system instead of staying resident until the kernel swaps it out under a
long prefill's pressure. A line in the log when it gave back 64 MiB or more.

Data sources register themselves from the engine process: vllm_ple_mmap keeps
lifetime counters (_STATS_LIFE, never reset — the log-window _STATS still
resets every VLLM_PLE_MMAP_STATS_SEC), mamba_utils_guarded pushes its guard
mirror every 512 steps, and the never-evict scheduler hook stores a Scheduler
reference whose BlockPool is read at refresh time.

The sidecar starts lazily, on the FIRST push from any source, and idempotently:
the engine process is the one that has data (the scheduler's __init__ alone
starts it at engine boot), whereas a frontend process that merely imports the
model module has nothing to report — if it grabbed the port first, every
metric would read 0 forever. Assumes the single-process deployment this fork
targets (one DGX Spark, no tensor parallelism); with several worker processes
the first one to push owns the endpoint. Every failure is logged once and
swallowed — telemetry must never take the server down.
"""
import errno
import logging
import os
import sys
import threading
import time

_PORT = int(os.environ.get("VLLM_CUSTOM_METRICS_PORT", "18400") or 0)
_INTERVAL = min(60.0, max(1.0, float(os.environ.get("VLLM_CUSTOM_METRICS_INTERVAL", "5") or 5)))
_IDLE_SEC = max(0.0, float(os.environ.get("VLLM_IDLE_EMPTY_CACHE_SEC", "0") or 0))
_IDLE_MIN = int(float(os.environ.get("VLLM_IDLE_EMPTY_CACHE_MIN_MB", "512") or 512) * 2**20)
_idle = {"since": None, "floor": None}   # idle start (monotonic s); reserved after the last release
_TRIM = os.environ.get("VLLM_IDLE_MALLOC_TRIM", "1") == "1"
_trim = {"done": False, "fn": None}      # malloc_trim ran in this idle period; the libc function

_started = False
_ensured = False
_lock = threading.Lock()

# Engine-side sources (same process as this module; see module docstring).
# Every push also kicks start() (lazy, idempotent) so the endpoint is bound by
# the process that actually has the data.
_guard_hits = [0]      # int, pushed by mamba_utils_guarded's telemetry refresh
_last_guard = [0]      # last value exported to Prometheus (counter delta)
_pin_source = [None]   # Scheduler, set by patch_never_evict's injected init
_last_life = {}        # last vllm_ple_mmap._STATS_LIFE snapshot (counter deltas)
_warned = [False]


def set_guard_hits(hits: int) -> None:
    _guard_hits[0] = int(hits)
    start()


def set_pin_source(scheduler) -> None:
    _pin_source[0] = scheduler
    start()


# Under "vllm." on purpose: vLLM configures handlers for the "vllm" logger
# tree only, so a logger named after this module ("vllm_custom_metrics") had
# no handler and its INFO lines were silently dropped in the server log.
_logger = logging.getLogger("vllm.custom_metrics")


def _has_handler(logger: logging.Logger) -> bool:
    lg = logger
    while lg is not None:
        if lg.handlers:
            return True
        if not lg.propagate:
            return False
        lg = lg.parent
    return False


def _emit(level: int, msg: str) -> None:
    """Log through vLLM's logger tree; fall back to stderr when nothing would
    actually print it (no vLLM logging configured, or level filtered out)."""
    if _has_handler(_logger) and _logger.isEnabledFor(level):
        _logger.log(level, "%s", msg)
    else:
        print(msg, file=sys.stderr, flush=True)


def _log_once(msg: str) -> None:
    if _warned[0]:
        return
    _warned[0] = True
    _emit(logging.WARNING, msg)


# --- metric objects ---------------------------------------------------------
_registry = None
_m = {}


def _ensure_metrics() -> None:
    """Create the registry and metric objects (idempotent; no threads)."""
    global _registry, _ensured
    if _ensured:
        return
    from prometheus_client import CollectorRegistry, Counter, Gauge

    _registry = CollectorRegistry()
    c = lambda name, doc: Counter(name, doc, registry=_registry)  # noqa: E731
    g = lambda name, doc: Gauge(name, doc, registry=_registry)  # noqa: E731
    _m.update({
        "ops": c("vllm:ple_mmap_ops", "PLE mmap lookup ops (engine-side, lifetime)."),
        "op_ms": c("vllm:ple_mmap_op_ms",
                   "PLE mmap lookup wall time in ms (hash+gather+H2D), lifetime. "
                   "rate(ple_mmap_op_ms_total)/rate(ple_mmap_ops_total) = ms/op."),
        "gather_ms": c("vllm:ple_mmap_gather_ms",
                       "PLE mmap disk-read time in ms, lifetime."),
        "rows": c("vllm:ple_mmap_rows", "PLE table rows gathered, lifetime."),
        "bytes": c("vllm:ple_mmap_bytes", "PLE table bytes read, lifetime."),
        "pf_hit": c("vllm:ple_mmap_prefetch_hit", "Batch-assembly prefetch hits."),
        "pf_miss": c("vllm:ple_mmap_prefetch_miss", "Batch-assembly prefetch misses."),
        "guard": c("vllm:mamba_state_copy_guard",
                   "Out-of-range mamba state copies skipped by the "
                   "bounds guard (tripwire; expected to stay 0 — any growth "
                   "is a new bug, it is also ERROR-logged). Sampled every "
                   "512 engine steps."),
        "pin_reserved": g("vllm:never_evict_blocks_reserved",
                              "KV blocks held by the never-evict pin."),
        "pin_queue": g("vllm:never_evict_pin_queue_blocks",
                           "KV blocks still in the never-evict pin queue "
                           "(rest of the reserved set is on loan in caches)."),
        "pin_bytes": g("vllm:never_evict_pin_bytes",
                           "Bytes pinned by the never-evict pin."),
        "torch_alloc": g("vllm:torch_cuda_allocated_bytes",
                         "Bytes in live tensors on the GPU (torch caching allocator, "
                         "engine process)."),
        "torch_res": g("vllm:torch_cuda_reserved_bytes",
                       "Bytes the torch caching allocator holds: live tensors, CUDA "
                       "graph pools and freed blocks kept for reuse. On GB10 this is "
                       "host memory, and it is not given back without empty_cache()."),
        "torch_peak": g("vllm:torch_cuda_max_reserved_bytes",
                        "Peak of vllm:torch_cuda_reserved_bytes since the engine started."),
        "idle_released": c("vllm:torch_idle_released_bytes",
                           "Bytes torch.cuda.empty_cache() gave back while the engine was "
                           "idle (VLLM_IDLE_EMPTY_CACHE_SEC)."),
    })
    _ensured = True


def _refresh_once() -> None:
    """Pull every source once and update the metrics. Never raises."""
    try:
        _ensure_metrics()
        # PLE mmap: lifetime counters -> prometheus counter deltas.
        import vllm_ple_mmap as _pm
        life = getattr(_pm, "_STATS_LIFE", None)
        if life:
            for key, metric in (("calls", "ops"), ("op_ms", "op_ms"),
                                ("gather_ms", "gather_ms"), ("rows", "rows"),
                                ("bytes", "bytes"), ("pf_hit", "pf_hit"),
                                ("pf_miss", "pf_miss")):
                now = life.get(key, 0)
                delta = now - _last_life.get(key, 0)
                if delta > 0:
                    _m[metric].inc(delta)
            _last_life.clear()
            _last_life.update(life)

        # Mamba state-copy guard tripwire (mirror refreshed every 512 steps;
        # monotonic, so it exports as a counter).
        if _guard_hits[0] > _last_guard[0]:
            _m["guard"].inc(_guard_hits[0] - _last_guard[0])
            _last_guard[0] = _guard_hits[0]

        # Never-evict pin: read the scheduler's BlockPool at refresh time.
        sched = _pin_source[0]
        pool = getattr(getattr(sched, "kv_cache_manager", None), "block_pool", None)
        get_pin_stats = getattr(pool, "get_pin_stats", None)
        if sched is not None and callable(get_pin_stats):
            reserved, held, per_group = get_pin_stats()
            _m["pin_reserved"].set(reserved)
            _m["pin_queue"].set(held)
            page_sizes = getattr(sched, "_pin_page_sizes", None)
            num_bytes = sum(count * page_sizes[gid]
                            for gid, count in per_group.items()) if page_sizes else 0
            _m["pin_bytes"].set(num_bytes)
    except Exception as exc:  # never let telemetry break anything
        _log_once(f"vllm_custom_metrics: refresh failed once and will keep "
                  f"retrying silently: {exc!r}")

    # torch's caching allocator, read from the refresher thread: allocator
    # statistics only (no CUDA call), and only once the engine has initialized
    # CUDA (a process that merely imported torch is left alone).
    try:
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_initialized():
            _m["torch_alloc"].set(torch.cuda.memory_allocated())
            _m["torch_res"].set(torch.cuda.memory_reserved())
            _m["torch_peak"].set(torch.cuda.max_memory_reserved())
            _maybe_release_idle(torch)
    except Exception as exc:
        _log_once(f"vllm_custom_metrics: torch allocator stats failed: {exc!r}")


def _maybe_release_idle(torch) -> None:
    """torch.cuda.empty_cache() once the engine has been idle _IDLE_SEC seconds
    (see the module docstring). Called from the refresher thread."""
    if _IDLE_SEC <= 0:
        return
    sched = _pin_source[0]
    has = getattr(sched, "has_requests", None) or getattr(sched, "has_unfinished_requests", None)
    if not callable(has):
        return  # no scheduler to ask: never release blind
    now = time.monotonic()
    if has():
        _idle["since"] = None
        _trim["done"] = False
        return
    if _idle["since"] is None:
        _idle["since"] = now
        return
    if now - _idle["since"] < _IDLE_SEC:
        return
    if _TRIM and not _trim["done"]:
        _trim["done"] = True
        _malloc_trim(now - _idle["since"])
    reserved = torch.cuda.memory_reserved()
    allocated = torch.cuda.memory_allocated()
    floor = _idle["floor"]
    if reserved - allocated < _IDLE_MIN or (floor is not None and reserved < floor + _IDLE_MIN):
        return
    torch.cuda.empty_cache()
    after = torch.cuda.memory_reserved()
    _idle["floor"] = after
    freed = max(0, reserved - after)
    if freed:
        _m["idle_released"].inc(freed)
    gib = 2.0 ** 30
    _emit(logging.INFO, "vllm_custom_metrics: engine idle %.0fs, torch.cuda.empty_cache() gave back "
          "%.2f GiB (reserved %.2f -> %.2f GiB, allocated %.2f GiB)"
          % (now - _idle["since"], freed / gib, reserved / gib, after / gib, allocated / gib))


def _rss() -> int:
    """This process's resident memory in bytes (CUDA allocations on GB10 are not in it)."""
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return 0


def _malloc_trim(idle_s: float) -> None:
    """glibc's malloc_trim(0): free heap pages back to the system. Logged when it gave
    back 64 MiB or more; skipped silently without glibc."""
    if _trim["fn"] is None:
        try:
            import ctypes
            _trim["fn"] = ctypes.CDLL("libc.so.6").malloc_trim
        except (OSError, AttributeError):
            _trim["fn"] = False
    if not _trim["fn"]:
        return
    before = _rss()
    _trim["fn"](0)
    after = _rss()
    if before - after >= 64 * 2**20:
        gib = 2.0 ** 30
        _emit(logging.INFO, "vllm_custom_metrics: engine idle %.0fs, malloc_trim(0) gave back %.2f GiB "
              "of heap (resident %.2f -> %.2f GiB)" % (idle_s, (before - after) / gib, before / gib, after / gib))


def _refresher() -> None:
    while True:
        _refresh_once()
        time.sleep(_INTERVAL)


def start() -> None:
    """Idempotently start the refresher thread and the HTTP endpoint."""
    global _started
    if _started or (_PORT <= 0 and _IDLE_SEC <= 0):
        return
    with _lock:
        if _started:
            return
        _started = True
        try:
            _ensure_metrics()
            from prometheus_client import start_wsgi_server
            threading.Thread(target=_refresher, name="vllm-custom-metrics",
                             daemon=True).start()
            if _IDLE_SEC > 0:
                _emit(logging.INFO, "vllm_custom_metrics: idle release on: torch.cuda.empty_cache() "
                      "after %.0fs without requests (at least %d MiB of freed blocks)"
                      % (_IDLE_SEC, _IDLE_MIN // 2**20))
            if _PORT <= 0:
                return
            start_wsgi_server(_PORT, addr="0.0.0.0", registry=_registry)
            _announce()
        except OSError as exc:
            if exc.errno == errno.EADDRINUSE:
                # Another process of this container got there first (expected
                # only with multi-process executors): not an error, but the
                # endpoint then reports that process's view.
                _log_once(f"vllm_custom_metrics: port {_PORT} already bound by "
                          f"another process; metrics are served from there")
            else:
                _log_once(f"vllm_custom_metrics: sidecar not started: {exc!r}")
        except Exception as exc:
            _log_once(f"vllm_custom_metrics: sidecar not started: {exc!r}")


def _announce() -> None:
    """Log that the sidecar is up. Never silent: if vLLM's logger is unusable,
    fall back to stderr — a sidecar that starts without saying so is exactly
    the failure mode this endpoint must not have."""
    _emit(logging.INFO, "vllm_custom_metrics: engine-side Prometheus sidecar on "
          ":%d (refresh every %.0fs)" % (_PORT, _INTERVAL))
