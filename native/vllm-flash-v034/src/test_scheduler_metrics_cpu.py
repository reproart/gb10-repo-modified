#!/usr/bin/env python3
"""CPU checks for patch_scheduler_metrics.py (no GPU, no vLLM; prometheus_client for 3):

    python3 src/test_scheduler_metrics_cpu.py

1. The patch on vLLM 0.34.0's own v1/core/sched/scheduler.py (src/testdata): the hook goes
   at the end, once; the file parses; a second run changes nothing; another file is refused.
2. The hook on a stand-in Scheduler: inert unless VLLM_CUSTOM_METRICS_PORT or
   VLLM_IDLE_EMPTY_CACHE_SEC is non-zero; then every Scheduler built hands itself to
   vllm_custom_metrics.set_pin_source after its own __init__; a failing sidecar is logged,
   never raised.
3. ../vllm-flash-ar/src/vllm_custom_metrics.py refreshes without its PLE module (absent on
   0.34) without logging a failure.
"""
import logging
import os
import shutil
import sys
import tempfile
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import patch_scheduler_metrics as pm

V034 = os.path.join(HERE, "testdata", "scheduler.py.v034")
ENV = ("VLLM_CUSTOM_METRICS_PORT", "VLLM_IDLE_EMPTY_CACHE_SEC")


class PatchTest(unittest.TestCase):
    def test_v034_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "scheduler.py")
            shutil.copy(V034, p)
            self.assertEqual(pm.patch(p), "INSTALLED")
            self.assertEqual(pm.patch(p), "already installed")
            with open(p) as f:
                self.assertEqual(f.read().count("Scheduler.__init__ = _gb10_init"), 1)
            with open(p, "w") as f:
                f.write("x = 1\n")
            with self.assertRaises(SystemExit):
                pm.patch(p)


class HookTest(unittest.TestCase):
    def build(self, env, sidecar=None):
        for k in ENV:
            os.environ.pop(k, None)
        os.environ.update(env)
        seen = []

        class Scheduler:
            def __init__(self, x):
                self.x = x

        fake = types.ModuleType("vllm_custom_metrics")
        fake.set_pin_source = sidecar or seen.append
        saved = sys.modules.get("vllm_custom_metrics")
        sys.modules["vllm_custom_metrics"] = fake
        ns = {"Scheduler": Scheduler, "logger": logging.getLogger("vllm.test_sched")}
        try:
            exec(compile(pm.HOOK, "hook", "exec"), ns)  # noqa: S102 - the hook as the patch appends it
            s = ns["Scheduler"](7)
        finally:
            for k in ENV:
                os.environ.pop(k, None)
            if saved is None:
                sys.modules.pop("vllm_custom_metrics", None)
            else:
                sys.modules["vllm_custom_metrics"] = saved
        return s, seen, ns

    def test_inert_by_default(self):
        s, seen, _ = self.build({"VLLM_CUSTOM_METRICS_PORT": "0", "VLLM_IDLE_EMPTY_CACHE_SEC": "0"})
        self.assertEqual(seen, [])
        self.assertEqual(s.x, 7)

    def test_port_or_idle_turns_it_on(self):
        for env in ({"VLLM_CUSTOM_METRICS_PORT": "18400"}, {"VLLM_IDLE_EMPTY_CACHE_SEC": "60"}):
            s, seen, _ = self.build(env)
            self.assertEqual(seen, [s])
            self.assertEqual(s.x, 7)

    def test_sidecar_failure_is_logged(self):
        def boom(_):
            raise RuntimeError("no port")

        with self.assertLogs("vllm.test_sched", "WARNING") as logs:
            s, _, _ = self.build({"VLLM_IDLE_EMPTY_CACHE_SEC": "60"}, sidecar=boom)
        self.assertEqual(s.x, 7)
        self.assertTrue(any("not started" in r for r in logs.output))


class ModuleTest(unittest.TestCase):
    def test_refresh_without_ple_module(self):
        try:
            import prometheus_client  # noqa: F401
        except ImportError:
            self.skipTest("needs prometheus_client")
        os.environ["VLLM_CUSTOM_METRICS_PORT"] = "0"
        sys.path.insert(0, os.path.join(HERE, "..", "..", "vllm-flash-ar", "src"))
        sys.modules.pop("vllm_ple_mmap", None)
        import vllm_custom_metrics as cm
        cm._warned[0] = False
        logger = logging.getLogger("vllm.custom_metrics")
        records = []
        handler = logging.Handler()
        handler.emit = records.append
        logger.addHandler(handler)
        try:
            cm._refresh_once()
        finally:
            logger.removeHandler(handler)
            os.environ.pop("VLLM_CUSTOM_METRICS_PORT", None)
        self.assertFalse(any("refresh failed" in r.getMessage() for r in records),
                         [r.getMessage() for r in records])


if __name__ == "__main__":
    unittest.main(verbosity=2)
