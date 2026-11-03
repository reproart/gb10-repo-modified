#!/usr/bin/env python3
"""CPU checks for patch_mtp_draft_vocab.py's private draft head (torch, numpy, safetensors;
no GPU, no vLLM: the Marlin calls are stand-ins that compute what Marlin computes from the
GPTQ layout they are given).

A small int8 GPTQ v1 lm_head (symmetric, g128) is written as a checkpoint; the patch's hook
runs against it with a draft vocabulary:
1. default: the int8 slice; the columns handed to Marlin are the head's own stored rows
   for the ids (padded to a multiple of 64 with the last id), the scales in the hidden
   states' dtype, the uint8b128 type; logits equal the full head's on the ids, -inf on
   the rest;
2. VLLM_MTP_DRAFT_HEAD=bf16: the bf16 slice, the same logits within bf16;
3. a head that is not symmetric int8 falls back to the bf16 slice and says why;
4. an unknown VLLM_MTP_DRAFT_HEAD is ignored with a warning; bf16 without a draft
   vocabulary is ignored and the hook stays out.

    python3 src/test_draft_head_cpu.py
"""
import ast
import json
import os
import sys
import tempfile
import types

import numpy as np
import torch
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
tree = ast.parse(open(os.path.join(HERE, "patch_mtp_draft_vocab.py")).read())
HOOK = next(n.value.value for n in tree.body
            if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "HOOK")

IN, VOCAB, GROUP = 256, 1000, 128
logs, calls = [], {}


class Log:
    def info(self, msg, *a):
        logs.append(("info", msg % a if a else msg))

    def warning(self, msg, *a):
        logs.append(("warning", msg % a if a else msg))


def dequant(qw, sc, zp=128):
    """GPTQ int8 v1 [in/4, n] int32 + scales [in/group, n] -> float32 [in, n]."""
    q = qw.contiguous().view(torch.uint8).reshape(qw.shape[0], qw.shape[1], 4)  # little-endian bytes
    q = q.permute(0, 2, 1).reshape(qw.shape[0] * 4, qw.shape[1]).float()
    return (q - zp) * sc.float().repeat_interleave(GROUP, 0)


def install_fakes():
    vllm = types.ModuleType("vllm")
    logger = types.ModuleType("vllm.logger")
    logger.init_logger = lambda name: Log()
    ops = types.ModuleType("vllm._custom_ops")

    def repack(q, perm, k, n, bits):
        calls["repack"] = (q.clone(), k, n, bits)
        return q

    ops.gptq_marlin_repack = repack
    st = types.ModuleType("vllm.scalar_type")
    st.scalar_types = types.SimpleNamespace(uint8b128="uint8b128", uint4b8="uint4b8")
    mu = types.ModuleType("vllm.model_executor.layers.quantization.utils.marlin_utils")
    mu.marlin_make_workspace_new = lambda dev: None

    def permute(s, k, n, group):
        calls["scales"] = (s.clone(), k, n, group)
        return s

    def apply(h, qw, sc, e1, e2, e3, ws, wtype, n, k, is_k_full):
        calls["wtype"] = wtype
        return (h.float() @ dequant(qw, sc)[:, :n]).to(h.dtype)

    mu.marlin_permute_scales = permute
    mu.apply_gptq_marlin_linear = apply
    vllm.logger, vllm._custom_ops, vllm.scalar_type = logger, ops, st
    sys.modules.update({"vllm": vllm, "vllm.logger": logger, "vllm._custom_ops": ops,
                        "vllm.scalar_type": st,
                        "vllm.model_executor.layers.quantization.utils.marlin_utils": mu})


def checkpoint(d, zp_stored=127):
    g = torch.Generator().manual_seed(0)
    q = torch.randint(0, 256, (IN, VOCAB), generator=g, dtype=torch.uint8)
    qw = q.reshape(IN // 4, 4, VOCAB).permute(0, 2, 1).contiguous().view(torch.int32).reshape(IN // 4, VOCAB)
    qz = torch.full((IN // GROUP, VOCAB), zp_stored, dtype=torch.uint8)
    qz = qz.reshape(IN // GROUP, VOCAB // 4, 4).contiguous().view(torch.int32).reshape(IN // GROUP, VOCAB // 4)
    sc = (torch.rand(IN // GROUP, VOCAB, generator=g) * 0.02 + 0.001).to(torch.float16)
    save_file({"lm_head.qweight": qw, "lm_head.qzeros": qz, "lm_head.scales": sc}, os.path.join(d, "head.safetensors"))
    json.dump({"weight_map": {n: "head.safetensors" for n in ("lm_head.qweight", "lm_head.qzeros", "lm_head.scales")}},
              open(os.path.join(d, "model.safetensors.index.json"), "w"))
    return qw, sc


class Processor:
    org_vocab_size = VOCAB

    def __init__(self, w):
        self.w = w

    def __call__(self, head, h):
        return (h.float() @ self.w).to(h.dtype)


def run(env, zp_stored=127):
    """Exec the hook with env; returns (installed, logits, ids, reference logits)."""
    logs.clear()
    calls.clear()
    for k in ("VLLM_MTP_DRAFT_VOCAB", "VLLM_MTP_DRAFT_HEAD"):
        os.environ.pop(k, None)
    os.environ.update(env)
    with tempfile.TemporaryDirectory() as d:
        qw, sc = checkpoint(d, zp_stored)
        os.environ["VLLM_MTP_DRAFT_VOCAB_CKPT"] = d
        if "VLLM_MTP_DRAFT_VOCAB" in env:
            np.save(env["VLLM_MTP_DRAFT_VOCAB"], np.array([5, 999, 17, 400, 1000, -3] + list(range(600, 900))))

        class Qwen3_8FlashNextMTP:
            def compute_logits(self, h, spec_step_idx=0):
                return None

        ns = {"torch": torch, "Qwen3_8FlashNextMTP": Qwen3_8FlashNextMTP}
        exec(compile(HOOK, "hook", "exec"), ns)
        installed = Qwen3_8FlashNextMTP.compute_logits is ns["_dv_compute_logits"]
        if not installed:
            return False, None, None, None
        m = Qwen3_8FlashNextMTP()
        m.lm_head = types.SimpleNamespace()  # GPTQ head: no dense .weight
        full = dequant(qw, sc, 128 if zp_stored == 127 else zp_stored + 1)
        m.logits_processor = Processor(full)
        h = torch.randn(3, IN, generator=torch.Generator().manual_seed(1)).to(torch.bfloat16)
        out = m.compute_logits(h)
        out2 = m.compute_logits(h)  # built once, reused
        assert torch.equal(out, out2)
        ids = sorted({5, 999, 17, 400} | set(range(600, 900)))
        return True, out, ids, (h.float() @ full)


fails = 0


def check(name, ok):
    global fails
    fails += not ok
    print(("PASS " if ok else "FAIL ") + name)


install_fakes()
tmp = tempfile.mkdtemp()
ids_file = os.path.join(tmp, "ids.npy")

installed, out, ids, ref = run({"VLLM_MTP_DRAFT_VOCAB": ids_file})
msg = [m for k, m in logs if "private draft head:" in m]
check("default with a draft vocabulary: the int8 slice", installed and bool(msg) and "int8 GPTQ-Marlin" in msg[0])
q, k, n, bits = calls["repack"]
order = [5, 999, 17, 400] + list(range(600, 900))
pad = order + [order[-1]] * (-len(order) % 64)
with tempfile.TemporaryDirectory() as d:
    qw_all, sc_all = checkpoint(d)
check("  Marlin gets the head's own stored rows for the ids, padded to a multiple of 64 with the last",
      n == len(pad) and n % 64 == 0 and k == IN and bits == 8 and torch.equal(q, qw_all[:, pad]))
s, _, _, group = calls["scales"]
check("  scales: those columns, in the hidden states' dtype, group 128",
      s.dtype == torch.bfloat16 and group == GROUP and torch.equal(s, sc_all[:, pad].to(torch.bfloat16)))
check("  the uint8b128 kernel type", calls.get("wtype") == "uint8b128")
on = out[:, ids].float()
rel = ((on - ref[:, ids]).norm() / ref[:, ids].norm()).item()
check(f"  logits on the ids = the full head's (rel {rel:.4f}), top-1 equal",
      rel < 1e-2 and torch.equal(on.argmax(1), ref[:, ids].argmax(1)))
rest = [i for i in range(VOCAB) if i not in set(ids)]
check("  every other id -inf, shape [T, vocab]", out.shape == (3, VOCAB) and bool(torch.isinf(out[:, rest]).all()))
check("  self-check logged", any("self-check: top-1 agreement 1.000" in m for _, m in logs))

installed, out_b, _, ref = run({"VLLM_MTP_DRAFT_VOCAB": ids_file, "VLLM_MTP_DRAFT_HEAD": "bf16"})
msg = [m for k, m in logs if "private draft head:" in m]
rel = ((out_b[:, ids].float() - ref[:, ids]).norm() / ref[:, ids].norm()).item()
check(f"DRAFT_HEAD=bf16: the bf16 slice (rel {rel:.4f})",
      installed and "repack" not in calls and bool(msg) and msg[0].startswith("MTP private draft head: bf16,")
      and rel < 1e-2)

installed, out_z, _, ref = run({"VLLM_MTP_DRAFT_VOCAB": ids_file}, zp_stored=126)
msg = [m for k, m in logs if "private draft head:" in m]
check("an asymmetric head (zero point 127): bf16 slice, and why",
      installed and "repack" not in calls and bool(msg) and "zero point 127" in msg[0])

installed, _, _, _ = run({"VLLM_MTP_DRAFT_VOCAB": ids_file, "VLLM_MTP_DRAFT_HEAD": "fp8"})
check("an unknown DRAFT_HEAD: warned, the default (int8 slice)",
      any(k == "warning" and "'fp8' ignored" in m for k, m in logs) and "repack" in calls)

installed, _, _, _ = run({"VLLM_MTP_DRAFT_HEAD": "bf16"})
check("bf16 without a draft vocabulary: warned, the hook stays out (the shared int8 head)",
      not installed and any(k == "warning" and "only applies" in m for k, m in logs))

installed, _, _, _ = run({})
check("nothing set: the hook stays out", not installed)

print("ALL OK" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
