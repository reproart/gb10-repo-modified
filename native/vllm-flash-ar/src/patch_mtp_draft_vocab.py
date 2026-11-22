#!/usr/bin/env python3
"""Build-time patch (qwen38-flash-dgx): private MTP draft head — reduced vocabulary, int8 or int4.

vLLM shares the target model's lm_head with the MTP draft (llm_base_proposer._maybe_share_lm_head),
so every draft step reads the whole head: 248,320 rows x 2560 int8 = 616 MiB per drafted token,
on a decode step that is memory-bandwidth bound. Two independent knobs shrink that read; the
target still verifies every drafted token, so outputs are identical to full-precision,
full-vocabulary drafting and only the acceptance rate can move:

  VLLM_MTP_DRAFT_VOCAB=<ids.npy>  the draft scores only those rows (a private slice of the head)
                                  and the logits of every other id are -inf. Acceptance drops for
                                  text whose tokens fall outside the set (e.g. CJK with an
                                  English/code-weighted set). Idea: upstream blazux / MiaAI-Lab.
                                  The slice keeps the head's own int8 GPTQ rows on the same
                                  GPTQ-Marlin kernel (65,536 rows: 160 MiB per draft step; the
                                  draft's logits on them are the target head's). Before
                                  2026-11-03 it was dequantized to bf16 (320 MiB):
  VLLM_MTP_DRAFT_HEAD=bf16        that bf16 slice, for comparison (needs DRAFT_VOCAB).
  VLLM_MTP_DRAFT_HEAD=int4        the draft uses a private int4 g128 RTN GPTQ-Marlin copy of the
                                  head (320 MB, built at first use, ~half the bytes of int8) over
                                  the FULL vocabulary, so every language keeps its acceptance.
                                  Combined with DRAFT_VOCAB the slice itself is int4 (~85 MB).

Fork note (Saren-Arterius/qwen3.8-Flash-DGX-AutoRound): this fork's lm_head is int8
GPTQ-Marlin, so there is no dense .weight to slice; rows are dequantized from the checkpoint's
lm_head.qweight/qzeros/scales (GPTQ v1, g128; VLLM_MTP_DRAFT_VOCAB_CKPT, default /model) in
16k-row chunks, then either kept as a bf16 slice or re-quantized to int4 with vLLM's own
marlin_quantize (chunked packing is bit-identical to whole-tensor packing). The int8 slice
takes the selected columns of qweight/scales as they are and repacks them for Marlin the way
vLLM loads the target's head (gptq_marlin_repack, marlin_permute_scales).

Two engines (vllm-flash-ar's pinned image, vllm-flash-v034's vLLM 0.34): the hook goes on the
file's one Qwen*MTP class (Qwen3_8FlashNextMTP, Qwen4ExpMTP) and reads the Marlin helpers'
signatures once, at the build: the pinned image's take a permutation, g_idx and act_order,
0.34's do not (no act-order there). The calls on the pinned image are the ones it always made.
On both the first draft logits come from the profiling run, eager, before any CUDA graph is
captured, so the head is built there; a first call inside a capture keeps the shared head for
that graph (correct, only slower) and says so.

usage: patch_mtp_draft_vocab.py <path to .../nvidia/mtp.py>
Inert unless VLLM_MTP_DRAFT_VOCAB or VLLM_MTP_DRAFT_HEAD is set at runtime.
"""
import ast
import re
import sys

MARK = "qwen38-flash-dgx: private MTP draft head"
HOOK = '''

# --- qwen38-flash-dgx: private MTP draft head (VLLM_MTP_DRAFT_VOCAB=<ids.npy>, VLLM_MTP_DRAFT_HEAD=int4|bf16)
import os as _dv_os
from vllm.logger import init_logger as _dv_init_logger

_dv_logger = _dv_init_logger(__name__)
_DV_CHUNK = 16384


def _dv_compute_logits(self, hidden_states: torch.Tensor, spec_step_idx: int = 0):
    st = getattr(self, "_dv_state", None)
    if st is None:
        if hidden_states.is_cuda and torch.cuda.is_current_stream_capturing():
            # building reads the checkpoint and copies to the GPU, which a capture cannot hold
            if not getattr(self, "_dv_capture_warned", False):
                self._dv_capture_warned = True
                _dv_logger.warning("MTP private draft head: first called inside a CUDA graph capture; "
                                   "that graph keeps the shared head")
            return _dv_shared_compute_logits(self, hidden_states, spec_step_idx)
        st = self._dv_state = _dv_build(self, hidden_states)
    ids, head, vocab = st
    red = head(hidden_states)
    if ids is None:
        return red[:, :vocab] if red.shape[1] != vocab else red
    full = red.new_full((red.shape[0], vocab), float("-inf"))
    full.index_copy_(1, ids, red)
    return full


def _dv_build(self, hidden_states):
    import time as _time
    import numpy as _np

    t0 = _time.time()
    dev = hidden_states.device
    head = self.lm_head  # the target's lm_head, shared in by the proposer
    w = getattr(head, "weight", None)
    dense = w is not None and w.dim() == 2
    vocab = int(getattr(self.logits_processor, "org_vocab_size", 0)
                or getattr(self.logits_processor, "vocab_size", 0)
                or (w.shape[0] if dense else head.org_vocab_size))
    if dense:
        out_f, in_f = w.shape
        rows = lambda sel: w.index_select(0, torch.as_tensor(sel, device=w.device)).to(torch.bfloat16)
        full_bytes = w.numel() * w.element_size()
        cols = None
    else:
        rows, out_f, in_f, full_bytes, cols = _dv_gptq_rows()

    ids_path = _dv_os.environ.get("VLLM_MTP_DRAFT_VOCAB")
    int4 = _dv_head_env == "int4"
    # a slice of an int8 GPTQ head stays int8 unless bf16 is asked for
    int8 = bool(ids_path) and _dv_head_env in ("", "int8") and cols is not None
    why = ""
    if int8:
        bits, zp, group = cols(None)
        if bits != 8 or zp != 128 or group not in (32, 64, 128) or in_f % 128:
            int8, why = False, " (head not symmetric int8 g32-128: %d-bit, zero point %d, group %d)" % (bits, zp, group)
    elif ids_path and _dv_head_env in ("", "int8"):
        why = " (the shared head has a dense weight: its own precision)"
    if ids_path:
        ids = _np.load(ids_path).astype(_np.int64)
        ids = ids[(ids >= 0) & (ids < vocab)]
        if (int4 or int8) and len(ids) % 64:  # Marlin needs N % 64 == 0; duplicates write identical values
            ids = _np.concatenate([ids, _np.repeat(ids[-1], 64 - len(ids) % 64)])
        sel = ids
    else:
        sel = _np.arange(out_f)
        ids = None
    n = len(sel)

    if int4:
        from vllm.scalar_type import scalar_types as _st
        from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new as _mws
        qws, scs = [], []
        for i in range(0, n, _DV_CHUNK):
            wt = rows(sel[i:i + _DV_CHUNK]).to(dev).t().contiguous()      # [in, chunk] bf16
            qw, s = _dv_quantize(wt, _st.uint4b8, 128)
            qws.append(qw); scs.append(s)
            del wt
        qw = torch.cat(qws, 1).contiguous(); sc = torch.cat(scs, 1).contiguous()
        del qws, scs
        apply = _dv_linear(qw, sc, _mws(dev), _st.uint4b8, n, in_f)
        bytes_ = qw.numel() * 4 + sc.numel() * sc.element_size()
        kind = "int4 g128 RTN GPTQ-Marlin"
    elif int8:
        from vllm.scalar_type import scalar_types as _st
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            marlin_make_workspace_new as _mws, marlin_permute_scales as _mps)
        qc, sc = cols(sel)                        # the head's own int8 rows: [in/4, n] int32, [in/group, n]
        qw = _dv_repack(qc.to(dev).contiguous(), in_f, n, 8)
        sc = _mps(sc.to(device=dev, dtype=hidden_states.dtype).contiguous(), in_f, n, group)
        del qc
        apply = _dv_linear(qw, sc, _mws(dev), _st.uint8b128, n, in_f)
        bytes_ = qw.numel() * 4 + sc.numel() * sc.element_size()
        kind = "int8 GPTQ-Marlin (the target head's own rows)"
    else:
        wk = torch.cat([rows(sel[i:i + _DV_CHUNK]).to(dev) for i in range(0, n, _DV_CHUNK)]).contiguous()

        def apply(h, wk=wk):
            return torch.nn.functional.linear(h.to(wk.dtype), wk)
        bytes_ = wk.numel() * wk.element_size()
        kind = "bf16" + why

    # self-check on the first real hidden states: private head vs the shared target head
    try:
        ref = self.logits_processor(self.lm_head, hidden_states)
        if ref is not None:
            ref = ref[:, :vocab].float()
            got = apply(hidden_states)[:, :ids.shape[0] if ids is not None else vocab].float()
            if ids is not None:
                ref = ref[:, torch.as_tensor(ids, device=ref.device)]
            else:
                got = got[:, :vocab]
            top1 = (ref.argmax(1) == got.argmax(1)).float().mean().item()
            rel = ((got - ref).norm() / ref.norm()).item()
            _dv_logger.info("MTP private draft head self-check: top-1 agreement %.3f, rel err %.4f (T=%d)",
                            top1, rel, ref.shape[0])
    except Exception as ex:  # never let a diagnostic kill the drafter
        _dv_logger.warning("MTP private draft head self-check skipped: %r", ex)

    _dv_logger.info("MTP private draft head: %s, %s of %d rows, %.0f -> %.0f MiB per draft step, built in %.1fs",
                    kind, ("%d" % ids.shape[0]) if ids is not None else "all", vocab,
                    full_bytes / 2**20, bytes_ / 2**20, _time.time() - t0)
    ids_t = torch.as_tensor(ids, device=dev) if ids is not None else None
    return ids_t, apply, vocab


def _dv_old_api(fn, name, n_required):
    """Whether a Marlin helper is the pinned image's: it has the act-order parameter (perm, g_idx,
    act_order) or that image's number of required ones (repack 5, apply 11, quantize 4; 0.34's
    take 4, 8, 3)."""
    import inspect as _inspect
    try:
        ps = _inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in ps or sum(p.default is p.empty for p in ps.values()) >= n_required


def _dv_repack(q, k, n, bits):
    from vllm import _custom_ops as _ops
    fn = _ops.gptq_marlin_repack
    if _dv_old_api(fn, "perm", 5):  # pinned image: an empty permutation (no act-order)
        return fn(q, torch.empty(0, dtype=torch.int, device=q.device), k, n, bits)
    return fn(q, k, n, bits)


def _dv_quantize(wt, wtype, group):
    """bf16 [in, n] -> Marlin-packed weight and permuted scales (RTN, symmetric)."""
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import marlin_quantize as fn
    out = fn(wt, wtype, group, False) if _dv_old_api(fn, "act_order", 4) else fn(wt, wtype, group)
    return out[1], out[2]


def _dv_linear(qw, sc, ws, wtype, n, k):
    """h -> h @ W^T on Marlin, W symmetric (an empty zero point), no act-order."""
    from vllm.model_executor.layers.quantization.utils.marlin_utils import apply_gptq_marlin_linear as fn
    e = torch.empty(0, dtype=torch.int, device=qw.device)
    if _dv_old_api(fn, "g_idx", 11):  # pinned image: zp, g_idx, g_idx_sort_indices empty; is_k_full
        return lambda h: fn(h.to(sc.dtype), qw, sc, e, e, e, ws, wtype, n, k, True)
    return lambda h: fn(h.to(sc.dtype), qw, sc, e, ws, wtype, n, k)


def _dv_gptq_rows():
    """Row dequantizer for the checkpoint's GPTQ lm_head (int8/int4 v1 layout: qweight [in/pack, out]
    packed along in, qzeros stores zp-1, scales [in/group, out]). rows(sel) -> bf16 [len(sel), in];
    cols(sel) -> those rows as stored (qweight int32 [in/pack, len(sel)], scales [in/group, len(sel)]),
    cols(None) -> (bits, zero point, group)."""
    import json as _json
    import numpy as _np
    from safetensors import safe_open as _safe_open

    ckpt = _dv_os.environ.get("VLLM_MTP_DRAFT_VOCAB_CKPT", "/model")
    wm = _json.load(open(_dv_os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
    names = ("lm_head.qweight", "lm_head.qzeros", "lm_head.scales")
    ts = {}
    for f in {wm[n] for n in names}:
        with _safe_open(_dv_os.path.join(ckpt, f), "pt", device="cpu") as sf:
            for n in names:
                if n in sf.keys():
                    ts[n] = sf.get_tensor(n)
    qw, qz, sc = (ts[n] for n in names)
    groups, out_f = sc.shape
    pack = out_f // qz.shape[1]
    bits = 32 // pack
    in_f = qw.shape[0] * pack
    group = in_f // groups
    mask = (1 << bits) - 1
    qwn = qw.numpy().view(_np.uint32)
    zp = int(qz.numpy().view(_np.uint32)[0, 0] & mask) + 1
    scn = sc.to(torch.float32).numpy()

    def rows(sel):
        q = qwn[:, sel]                                                  # [in/pack, k]
        nib = _np.stack([(q >> (i * bits)) & mask for i in range(pack)], axis=1).reshape(in_f, len(sel))
        wf = (nib.astype(_np.float32) - zp).reshape(groups, group, len(sel)) * scn[:, None, sel]
        return torch.from_numpy(_np.ascontiguousarray(wf.reshape(in_f, len(sel)).T)).to(torch.bfloat16)

    def cols(sel):
        if sel is None:
            return bits, zp, group
        return (torch.from_numpy(_np.ascontiguousarray(qwn[:, sel]).view(_np.int32)),
                sc[:, torch.as_tensor(sel, dtype=torch.long)])
    return rows, out_f, in_f, qw.numel() * 4 + sc.numel() * sc.element_size(), cols


_dv_head_env = _dv_os.environ.get("VLLM_MTP_DRAFT_HEAD", "").lower()
if _dv_head_env not in ("", "int4", "int8", "bf16"):
    _dv_logger.warning("VLLM_MTP_DRAFT_HEAD=%r ignored (int8, int4 or bf16)", _dv_head_env)
    _dv_head_env = ""
if _dv_head_env == "bf16" and not _dv_os.environ.get("VLLM_MTP_DRAFT_VOCAB"):
    # a full-vocabulary bf16 copy of the head (~1.2 GiB) would be slower than sharing the int8 one
    _dv_logger.warning("VLLM_MTP_DRAFT_HEAD=bf16 ignored: it only applies to a VLLM_MTP_DRAFT_VOCAB slice")
    _dv_head_env = ""
_dv_shared_compute_logits = Qwen3_8FlashNextMTP.compute_logits
if _dv_os.environ.get("VLLM_MTP_DRAFT_VOCAB") or _dv_head_env == "int4":
    Qwen3_8FlashNextMTP.compute_logits = _dv_compute_logits  # type: ignore[method-assign]
    _dv_logger.info("MTP private draft head enabled: vocab=%s head=%s",
                    _dv_os.environ.get("VLLM_MTP_DRAFT_VOCAB"), _dv_os.environ.get("VLLM_MTP_DRAFT_HEAD", "shared"))
'''

HOOK_CLASS = "Qwen3_8FlashNextMTP"   # the name the hook is written with


def mtp_class(src):
    """The file's one MTP model class: Qwen3_8FlashNextMTP (pinned image), Qwen4ExpMTP (0.34)."""
    found = re.findall(r"^class (Qwen\w*MTP)\(", src, re.MULTILINE)
    if len(found) != 1:
        sys.exit("patch_mtp_draft_vocab.py: expected one Qwen*MTP class, found %s" % (found or "none"))
    return found[0]


def hook_for(cls):
    return HOOK.replace(HOOK_CLASS, cls)


def patch(path):
    with open(path) as f:
        src = f.read()
    if MARK in src:
        return "already installed"
    cls = mtp_class(src)
    out = src.rstrip("\n") + hook_for(cls)
    ast.parse(out)
    with open(path, "w") as f:
        f.write(out)
    return "INSTALLED on " + cls


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: patch_mtp_draft_vocab.py <mtp.py>")
    print("  draft-head hook", patch(sys.argv[1]), "in", sys.argv[1],
          "(inert unless VLLM_MTP_DRAFT_VOCAB / VLLM_MTP_DRAFT_HEAD is set)")
