#!/usr/bin/env python3
"""Build-time patch: the engine-side metrics sidecar and the idle release on vLLM 0.34.

../vllm-flash-ar/src/vllm_custom_metrics.py (its patch 13), the same module, copied next to
vLLM by the Dockerfile: a refresher thread in the engine process that serves torch's
caching-allocator gauges on its own Prometheus port (VLLM_CUSTOM_METRICS_PORT) and, after
VLLM_IDLE_EMPTY_CACHE_SEC seconds with the scheduler empty, gives the cached GPU blocks back
(torch.cuda.empty_cache(); on GB10 they are host memory) and runs malloc_trim(0) once. The
dev image starts it from its never-evict scheduler hook; here a hook on 0.34's
Scheduler.__init__ hands it the scheduler (has_requests() is what the idle check asks). The
dev image's other sources (the PLE mmap counters, the mamba guard, the pin gauges) do not
exist on 0.34 and are skipped.

Inert unless VLLM_CUSTOM_METRICS_PORT or VLLM_IDLE_EMPTY_CACHE_SEC is set (non-zero).
usage: patch_scheduler_metrics.py <path to vllm/v1/core/sched/scheduler.py>
"""
import ast
import sys

MARK = "gb10: engine metrics + idle release"
ANCHOR = "class Scheduler(SchedulerInterface):"
HOOK = '''

# --- gb10: engine metrics + idle release (vllm-flash-v034/src/patch_scheduler_metrics.py)
import os as _gb10_os


def _gb10_metrics_wanted(env=_gb10_os.environ):
    def num(name):
        try:
            return float(env.get(name, "0") or 0)
        except ValueError:
            return 0.0
    return num("VLLM_CUSTOM_METRICS_PORT") > 0 or num("VLLM_IDLE_EMPTY_CACHE_SEC") > 0


if _gb10_metrics_wanted():
    _gb10_scheduler_init = Scheduler.__init__

    def _gb10_init(self, *args, **kwargs):
        _gb10_scheduler_init(self, *args, **kwargs)
        try:
            import vllm_custom_metrics

            vllm_custom_metrics.set_pin_source(self)  # starts the sidecar in this (the engine's) process
        except Exception as exc:  # noqa: BLE001 - telemetry never takes the server down
            logger.warning("gb10 engine metrics not started: %r", exc)

    Scheduler.__init__ = _gb10_init
'''


def patch(path):
    with open(path) as f:
        src = f.read()
    if MARK in src:
        return "already installed"
    if src.count(ANCHOR) != 1 or "\nlogger = " not in src:
        raise SystemExit(f"patch_scheduler_metrics: {path}: no single {ANCHOR!r} or no module logger: "
                         "not the scheduler.py this was written for")
    out = src.rstrip("\n") + HOOK
    ast.parse(out)
    with open(path, "w") as f:
        f.write(out)
    return "INSTALLED"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: patch_scheduler_metrics.py <scheduler.py>")
    print("  engine metrics hook", patch(sys.argv[1]), "in", sys.argv[1],
          "(inert unless VLLM_CUSTOM_METRICS_PORT or VLLM_IDLE_EMPTY_CACHE_SEC is set)")
