"""CPU test for vllm_custom_metrics: sources -> refreshed gauges/counters.

No network and no vLLM: start() is never called (port 0), the PLE module is a
fake injected into sys.modules, and the exposition is generated straight from
the registry. Checks that every metric appears with the right value and that
counters accumulate by DELTA across refreshes (not by absolute value).
"""
import os
import sys
import types

os.environ["VLLM_CUSTOM_METRICS_PORT"] = "0"  # never start the HTTP server here
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import vllm_custom_metrics as cm  # noqa: E402

# Fake PLE source: lifetime counters, as the real vllm_ple_mmap._STATS_LIFE.
ple = types.ModuleType("vllm_ple_mmap")
ple._STATS_LIFE = {
    "calls": 100, "op_ms": 250.0, "gather_ms": 120.0,
    "rows": 1000, "bytes": 160_000, "pf_hit": 7, "pf_miss": 3,
}
sys.modules["vllm_ple_mmap"] = ple


# Fake never-evict source: a Scheduler-ish object with a BlockPool-ish pool.
class _Pool:
    def get_pin_stats(self):
        return 5, 2, {0: 3, 1: 2}  # reserved, in-queue, per-group blocks


class _Sched:
    kv_cache_manager = types.SimpleNamespace(block_pool=_Pool())
    _pin_page_sizes = [160, 3200]


# Fake torch with an initialized CUDA allocator (the real one needs a GPU).
_fake_torch = types.ModuleType("torch")
_fake_torch.cuda = types.SimpleNamespace(
    is_initialized=lambda: True,
    memory_allocated=lambda: 90_000_000,
    memory_reserved=lambda: 104_000_000,
    max_memory_reserved=lambda: 112_000_000,
)
_real_torch = sys.modules.get("torch")
sys.modules["torch"] = _fake_torch

cm.set_guard_hits(4)
cm.set_pin_source(_Sched())
cm._refresh_once()

from prometheus_client import generate_latest  # noqa: E402

text = generate_latest(cm._registry).decode()
expected = [
    ("vllm:ple_mmap_ops_total", 100.0),
    ("vllm:ple_mmap_op_ms_total", 250.0),
    ("vllm:ple_mmap_gather_ms_total", 120.0),
    ("vllm:ple_mmap_rows_total", 1000.0),
    ("vllm:ple_mmap_bytes_total", 160_000.0),
    ("vllm:ple_mmap_prefetch_hit_total", 7.0),
    ("vllm:ple_mmap_prefetch_miss_total", 3.0),
    ("vllm:mamba_state_copy_guard_total", 4.0),
    ("vllm:never_evict_blocks_reserved", 5.0),
    ("vllm:never_evict_pin_queue_blocks", 2.0),
    ("vllm:never_evict_pin_bytes", 3 * 160 + 2 * 3200),
    ("vllm:torch_cuda_allocated_bytes", "9e+07"),  # prometheus formatting
    ("vllm:torch_cuda_reserved_bytes", "1.04e+08"),
    ("vllm:torch_cuda_max_reserved_bytes", "1.12e+08"),
]
for name, val in expected:
    assert f"{name} {val}" in text, f"{name} {val} not in exposition:\n{text}"
print("first refresh: all", len(expected), "metrics present with expected values")

# Second refresh: counters must grow by the DELTA only (100 -> 150 = +50),
# not by the new absolute value.
ple._STATS_LIFE["calls"] = 150
cm._refresh_once()
text = generate_latest(cm._registry).decode()
assert "vllm:ple_mmap_ops_total 150.0" in text, text
assert "vllm:ple_mmap_op_ms_total 250.0" in text  # unchanged source -> unchanged
print("second refresh: delta accounting OK (100 -> 150)")

# A source that disappears (e.g. no PLE module) must not raise: the refresh
# reads getattr(..., None) paths. Simulate by dropping _STATS_LIFE.
del ple._STATS_LIFE
cm._refresh_once()
print("missing-source refresh: no exception")

# CUDA not initialized (a frontend process that imported torch): the torch
# gauges keep their last value, the allocator is not touched.
_fake_torch.cuda.is_initialized = lambda: False
_fake_torch.cuda.memory_reserved = lambda: (_ for _ in ()).throw(AssertionError("touched"))
cm._refresh_once()
text = generate_latest(cm._registry).decode()
assert "vllm:torch_cuda_reserved_bytes 1.04e+08" in text, text
if _real_torch is not None:
    sys.modules["torch"] = _real_torch
else:
    del sys.modules["torch"]
print("torch allocator gauges: set while CUDA is up, left alone before")

# set_pin_source(None): pin gauges keep their last value, nothing raises.
cm.set_pin_source(None)
cm._refresh_once()
text = generate_latest(cm._registry).decode()
assert "vllm:never_evict_blocks_reserved 5.0" in text, text
print("pin source detached: gauges hold last value, no exception")

# _announce must NEVER be silent: a sidecar that starts without saying so is
# the failure mode this endpoint exists to avoid.
import contextlib  # noqa: E402
import io  # noqa: E402
import logging  # noqa: E402

# 1) No logging configured at all (no handler anywhere): stderr fallback.
_buf = io.StringIO()
_root_handlers = logging.getLogger().handlers[:]
logging.getLogger().handlers.clear()
try:
    with contextlib.redirect_stderr(_buf):
        cm._announce()
finally:
    logging.getLogger().handlers[:] = _root_handlers
assert "sidecar on" in _buf.getvalue(), f"announce was silent: {_buf.getvalue()!r}"
print("announce: non-silent fallback OK")

# 2) vLLM-style setup: a handler on the "vllm" logger tree only, propagation
# off. The announce must land there (it used to go to a logger outside the
# tree and vanish from the server log).
_vl = logging.getLogger("vllm")
_cap = io.StringIO()
_h = logging.StreamHandler(_cap)
_vl.addHandler(_h)
_vl.setLevel(logging.INFO)
_vl.propagate = False
try:
    cm._announce()
finally:
    _vl.removeHandler(_h)
    _vl.propagate = True
    _vl.setLevel(logging.NOTSET)
assert "sidecar on" in _cap.getvalue(), f"announce missed the vllm logger: {_cap.getvalue()!r}"
print("announce: reaches the vllm logger tree OK")

# Idle release: torch.cuda.empty_cache() after VLLM_IDLE_EMPTY_CACHE_SEC without
# requests, once per rise of the allocator's hold; never while busy, never blind.
GiB = 2**30
alloc = {"reserved": 108 * GiB, "allocated": 100 * GiB, "graph_pool": 1 * GiB, "calls": 0}


def _empty_cache():
    alloc["calls"] += 1
    alloc["reserved"] = alloc["allocated"] + alloc["graph_pool"]  # freed blocks go, graph pools stay


_t = types.ModuleType("torch")
_t.cuda = types.SimpleNamespace(
    is_initialized=lambda: True, memory_allocated=lambda: alloc["allocated"],
    memory_reserved=lambda: alloc["reserved"], max_memory_reserved=lambda: 108 * GiB,
    empty_cache=_empty_cache)
sys.modules["torch"] = _t
clock = [1000.0]
cm.time = types.SimpleNamespace(monotonic=lambda: clock[0], sleep=lambda s: None)
busy = [True]


class _IdleSched(_Sched):
    def has_requests(self):
        return busy[0]


cm.set_pin_source(_IdleSched())
cm._IDLE_SEC, cm._IDLE_MIN = 60.0, 512 * 2**20
logs = []
cm._emit = lambda level, msg: logs.append(msg)


def tick(seconds=5):
    clock[0] += seconds
    cm._refresh_once()


tick(); tick()
assert alloc["calls"] == 0, "released while busy"
busy[0] = False
tick()                      # idle starts here
for _ in range(11):         # 55 s idle: not yet
    tick()
assert alloc["calls"] == 0, "released before VLLM_IDLE_EMPTY_CACHE_SEC"
tick()                      # 60 s
assert alloc["calls"] == 1 and alloc["reserved"] == 101 * GiB, alloc
assert any("gave back 7.00 GiB" in m for m in logs), logs
text = generate_latest(cm._registry).decode()
assert f"vllm:torch_idle_released_bytes_total {float(7 * GiB)}" in text or "vllm:torch_idle_released_bytes_total 7.516192768e+09" in text, text
for _ in range(20):         # still idle: the graph pool (1 GiB > 512 MiB) must not make it repeat
    tick()
assert alloc["calls"] == 1, "repeated on blocks empty_cache cannot free"
busy[0] = True; tick()      # new work raises the hold by 3 GiB
alloc["reserved"] += 3 * GiB
busy[0] = False
for _ in range(14):
    tick()
assert alloc["calls"] == 2 and alloc["reserved"] == 101 * GiB, alloc
cm.set_pin_source(None)     # no scheduler to ask: no release
alloc["reserved"] += 5 * GiB
for _ in range(30):
    tick()
assert alloc["calls"] == 2, "released without a scheduler"
cm._IDLE_SEC = 0.0
cm.set_pin_source(_IdleSched())
for _ in range(30):
    tick()
assert alloc["calls"] == 2, "released with the switch off"
if _real_torch is not None:
    sys.modules["torch"] = _real_torch
else:
    del sys.modules["torch"]
print("idle release: after 60 s idle, once per rise, never busy / blind / off")

print("ALL OK")
