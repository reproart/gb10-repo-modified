#!/usr/bin/env python3
"""CPU checks for patch_mtp_draft_vocab.py's private draft head (torch, numpy, safetensors;
no GPU, no vLLM: the Marlin calls are stand-ins with the real helpers' signatures that
compute what Marlin computes from the layout they are given).

A small int8 GPTQ v1 lm_head (symmetric, g128) is written as a checkpoint; the patch's hook
runs against it with a draft vocabulary, on both engines' Marlin helpers (the pinned
image's: a permutation, g_idx, act_order; vLLM 0.34's: none of them; and the pinned
shapes under other parameter names, told apart by their number of required arguments):
1. default: the int8 slice; the columns handed to Marlin are the head's own stored rows
   for the ids (padded to a multiple of 64 with the last id), the scales in the hidden
   states' dtype, the uint8b128 type; logits equal the full head's on the ids, -inf on
   the rest;
2. VLLM_MTP_DRAFT_HEAD=bf16: the bf16 slice, the same logits within bf16;
3. VLLM_MTP_DRAFT_HEAD=int4: the full vocabulary re-quantized to int4 in chunks;
4. a head that is not symmetric int8 falls back to the bf16 slice and says why;
5. an unknown VLLM_MTP_DRAFT_HEAD is ignored with a warning; bf16 without a draft
   vocabulary is ignored and the hook stays out.
Then the patch step: the pinned image's mtp.py (src/testdata/mtp.py.image without its hook)
and 0.34's (../vllm-flash-v034/src/testdata/mtp.py.v034) each get the hook on their own
MTP class, once; a file without one class is refused.

    python3 src/test_draft_head_cpu.py
"""
import ast
import json
import os
import shutil
import sys
import tempfile
import types

import numpy as np
import torch
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import patch_mtp_draft_vocab as patcher  # noqa: E402

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


def matmul(h, qw, sc, wtype, n):
    """What the stand-in Marlin computes: int8 = GPTQ layout as handed over; int4 = the stand-in
    quantizer's unpacked values [in, n] (4-bit, bias 8) and scales [in/128, n]."""
    if wtype == "uint4b8":
        w = (qw.float() - 8) * sc.float().repeat_interleave(128, 0)
    else:
        w = dequant(qw, sc)
    return (h.float() @ w[:, :n]).to(h.dtype)


def fake_quantize(w, quant_type, group_size):
    """Symmetric RTN int4 per group; returns [w_ref, q, s] like vLLM 0.34's marlin_quantize."""
    calls.setdefault("quantize", []).append(tuple(w.shape))
    g = w.float().reshape(-1, group_size, w.shape[1])
    s = g.abs().amax(1) / 7
    q = (g / s[:, None]).round().clamp(-8, 7) + 8
    return [w, q.reshape(w.shape).to(torch.int32), s.to(w.dtype)]


# The helpers' parameters as each engine declares them (only the names are read).
def pinned_repack(b_q_weight, perm, size_k, size_n, num_bits, is_a_8bit=False):
    calls["repack"] = (b_q_weight.clone(), size_k, size_n, num_bits)
    calls["perm"] = perm
    return b_q_weight


def v034_repack(b_q_weight, size_k, size_n, num_bits):
    calls["repack"] = (b_q_weight.clone(), size_k, size_n, num_bits)
    return b_q_weight


def pinned_apply(input, weight, weight_scale, weight_zp, g_idx, g_idx_sort_indices, workspace, wtype,
                 output_size_per_partition, input_size_per_partition, is_k_full, input_global_scale=None,
                 bias=None, use_fp32_reduce=True, input_dtype=None):
    calls["wtype"], calls["is_k_full"] = wtype, is_k_full
    calls["empty"] = [t.numel() for t in (weight_zp, g_idx, g_idx_sort_indices)]
    return matmul(input, weight, weight_scale, wtype, output_size_per_partition)


def v034_apply(input, weight, weight_scale, weight_zp, workspace, wtype, output_size_per_partition,
               input_size_per_partition, bias=None, use_fp32_reduce=True):
    calls["wtype"], calls["empty"] = wtype, [weight_zp.numel()]
    assert input_size_per_partition == input.shape[-1]
    return matmul(input, weight, weight_scale, wtype, output_size_per_partition)


def pinned_quantize(w, quant_type, group_size, act_order, test_perm=None, input_dtype=None):
    assert act_order is False
    w_ref, q, s = fake_quantize(w, quant_type, group_size)
    return w_ref, q, s, None, None, None


def v034_quantize(w, quant_type, group_size):
    return fake_quantize(w, quant_type, group_size)


def renamed_repack(q, p, k, n, b, is_a_8bit=False):          # the pinned shapes, other names
    return pinned_repack(q, p, k, n, b)


def renamed_apply(x, w, s, zp, gi, gs, ws, t, n, k, full, bias=None):
    return pinned_apply(x, w, s, zp, gi, gs, ws, t, n, k, full)


def renamed_quantize(w, t, g, ao, test_perm=None):
    return pinned_quantize(w, t, g, ao)


ENGINES = {"pinned": (pinned_repack, pinned_apply, pinned_quantize),
           "v034": (v034_repack, v034_apply, v034_quantize),
           "renamed": (renamed_repack, renamed_apply, renamed_quantize)}


def install_fakes(engine):
    repack, apply, quantize = ENGINES[engine]
    vllm = types.ModuleType("vllm")
    logger = types.ModuleType("vllm.logger")
    logger.init_logger = lambda name: Log()
    ops = types.ModuleType("vllm._custom_ops")
    ops.gptq_marlin_repack = repack
    st = types.ModuleType("vllm.scalar_type")
    st.scalar_types = types.SimpleNamespace(uint8b128="uint8b128", uint4b8="uint4b8")
    mu = types.ModuleType("vllm.model_executor.layers.quantization.utils.marlin_utils")
    mu.marlin_make_workspace_new = lambda dev: None

    def permute(s, k, n, group):
        calls["scales"] = (s.clone(), k, n, group)
        return s

    mu.marlin_permute_scales = permute
    mu.apply_gptq_marlin_linear = apply
    mt = types.ModuleType("vllm.model_executor.layers.quantization.utils.marlin_utils_test")
    mt.marlin_quantize = quantize
    vllm.logger, vllm._custom_ops, vllm.scalar_type = logger, ops, st
    sys.modules.update({"vllm": vllm, "vllm.logger": logger, "vllm._custom_ops": ops,
                        "vllm.scalar_type": st,
                        "vllm.model_executor.layers.quantization.utils.marlin_utils": mu,
                        "vllm.model_executor.layers.quantization.utils.marlin_utils_test": mt})


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


def run(env, zp_stored=127, engine="pinned"):
    """Exec the hook with env on an engine's Marlin helpers and MTP class name; returns
    (installed, logits, ids, reference logits)."""
    install_fakes(engine)
    cls_name = {"pinned": "Qwen3_8FlashNextMTP", "v034": "Qwen4ExpMTP", "renamed": "Qwen3_8FlashNextMTP"}[engine]
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

        def shared(self, h, spec_step_idx=0):
            return None

        mtp = type(cls_name, (), {"compute_logits": shared})
        ns = {"torch": torch, cls_name: mtp}
        exec(compile(patcher.hook_for(cls_name), "hook", "exec"), ns)
        assert ns["_dv_shared_compute_logits"] is shared   # kept for a first call inside a capture
        installed = mtp.compute_logits is ns["_dv_compute_logits"]
        if not installed:
            return False, None, None, None
        m = mtp()
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


tmp = tempfile.mkdtemp()
ids_file = os.path.join(tmp, "ids.npy")
order = [5, 999, 17, 400] + list(range(600, 900))
pad = order + [order[-1]] * (-len(order) % 64)
with tempfile.TemporaryDirectory() as d:
    qw_all, sc_all = checkpoint(d)

for engine in ("pinned", "v034", "renamed"):
    installed, out, ids, ref = run({"VLLM_MTP_DRAFT_VOCAB": ids_file}, engine=engine)
    msg = [m for k, m in logs if "private draft head:" in m]
    check(f"[{engine}] default with a draft vocabulary: the int8 slice",
          installed and bool(msg) and "int8 GPTQ-Marlin" in msg[0])
    q, k, n, bits = calls["repack"]
    check("  Marlin gets the head's own stored rows for the ids, padded to a multiple of 64 with the last",
          n == len(pad) and n % 64 == 0 and k == IN and bits == 8 and torch.equal(q, qw_all[:, pad]))
    if engine in ("pinned", "renamed"):
        check(f"  {engine} helpers: an empty permutation; zp, g_idx, sort indices empty; is_k_full",
              calls["perm"].numel() == 0 and calls["empty"] == [0, 0, 0] and calls["is_k_full"] is True)
    else:
        check("  0.34 helpers: no permutation, an empty zero point", "perm" not in calls and calls["empty"] == [0])
    s, _, _, group = calls["scales"]
    check("  scales: those columns, in the hidden states' dtype, group 128",
          s.dtype == torch.bfloat16 and group == GROUP and torch.equal(s, sc_all[:, pad].to(torch.bfloat16)))
    check("  the uint8b128 kernel type", calls.get("wtype") == "uint8b128")
    on = out[:, ids].float()
    rel = ((on - ref[:, ids]).norm() / ref[:, ids].norm()).item()
    check(f"  logits on the ids = the full head's (rel {rel:.4f}), top-1 equal",
          rel < 1e-2 and torch.equal(on.argmax(1), ref[:, ids].argmax(1)))
    rest = [i for i in range(VOCAB) if i not in set(ids)]
    check("  every other id -inf, shape [T, vocab]",
          out.shape == (3, VOCAB) and bool(torch.isinf(out[:, rest]).all()))
    check("  self-check logged", any("self-check: top-1 agreement 1.000" in m for _, m in logs))

    installed, out4, _, ref = run({"VLLM_MTP_DRAFT_HEAD": "int4"}, engine=engine)
    msg = [m for k, m in logs if "private draft head:" in m]
    rel = ((out4.float() - ref).norm() / ref.norm()).item()
    check(f"[{engine}] DRAFT_HEAD=int4: the full vocabulary re-quantized (rel {rel:.3f}), uint4b8",
          installed and bool(msg) and "int4 g128" in msg[0] and "all of 1000" in msg[0]
          and calls.get("wtype") == "uint4b8" and calls["quantize"] == [(IN, VOCAB)]
          and out4.shape == (3, VOCAB) and rel < 0.15)

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

# The patch step, on each engine's own mtp.py.
V034 = os.path.join(HERE, "..", "..", "vllm-flash-v034", "src", "testdata", "mtp.py.v034")
with open(os.path.join(HERE, "testdata", "mtp.py.image")) as f:   # the image's file, its hook taken off
    pinned_src = f.read().split("\n\n# --- " + patcher.MARK)[0] + "\n"
for name, cls, other in (("pinned", "Qwen3_8FlashNextMTP", "Qwen4ExpMTP"),
                         ("v034", "Qwen4ExpMTP", "Qwen3_8FlashNextMTP")):
    p = os.path.join(tmp, "mtp_%s.py" % name)
    if name == "pinned":
        with open(p, "w") as f:
            f.write(pinned_src)
    else:
        shutil.copy(V034, p)
    first, second = patcher.patch(p), patcher.patch(p)
    with open(p) as f:
        src = f.read()
    ast.parse(src)
    check(f"[{name}] patch: the hook on {cls}, once",
          first == "INSTALLED on " + cls and second == "already installed" and src.count(patcher.MARK) == 1
          and f"    {cls}.compute_logits = _dv_compute_logits" in src
          and f"_dv_shared_compute_logits = {cls}.compute_logits" in src and other not in src[src.index(patcher.MARK):])
p = os.path.join(tmp, "none.py")
with open(p, "w") as f:
    f.write("class Foo:\n    pass\n")
try:
    patcher.patch(p)
    refused = False
except SystemExit as ex:
    refused = "found none" in str(ex)
check("a file without a Qwen*MTP class: refused", refused)
shutil.rmtree(tmp)

print("ALL OK" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
