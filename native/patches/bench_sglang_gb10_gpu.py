#!/usr/bin/env python3
"""GPU measurements for the Qwen3.8-Flash-Next SGLang profile, in its venv, with the
server stopped (it needs ~3 GB and a few minutes):

    ~/spark/venv-sglang-0.5.21/bin/python patches/bench_sglang_gb10_gpu.py

1. hyper-connection mix at decode sizes (1-32 rows): SGLang's BF16 kernel against
   the FP8 copy (gb10_fp8_hc_kernel), each call on another copy of the weights so
   they come from memory as in a decode step (~98 different mixes a step);
   correctness against the same math in FP32.
2. the FP8 copy's tiles at 4 and 32 rows (a verify at 1 and 8 requests): which
   (BLOCK_N, BLOCK_K, BLOCK_J, BLOCK_R, warps, stages) reads the weights fastest,
   against the default.
3. the mix at prefill sizes (2048-8192 rows, eager): the eager chain FP8_HC used
   until now, the new path (cuBLAS + one Triton pass for the gate), and SGLang's
   own torch.compile'd BF16 chain (what runs without FP8_HC).
4. the FP8 side layers (GDN, attention, shared expert) at prefill sizes: the FP8
   Marlin GEMM they run on now (weight-only, made for few rows) against cuBLAS on
   BF16 weights (stock SGLang), dequantize + cuBLAS, and FP8 W8A8 with per-token
   activation scales (SGLang's fp8_scaled_mm, its Triton scaled_mm,
   torch._scaled_mm), the server's FP8_SIDE_PREFILL=w8a8 path (the plain weight
   restored from Marlin's layout each call, then fp8_scaled_mm: "marlin->w8a8"),
   and what each would cost a 4096-token chunk over the model's 36 GDN +
   12 attention + 48 shared-expert layers.
5. Qwen3.8-27B (RadixArk NVFP4 + DFlash2, models/qwen3.8-27b.sh) decode GEMMs at
   verify sizes, 15 / 60 / 120 / 240 rows (draft 15 at 1 / 4 / 8 / 16 streams):
   the checkpoint's FP8 per-tensor layers (GDN out_proj and attention o_proj,
   GDN in_proj_qkvz, attention qkv_proj) on SGLang's path (static activation
   quant + FlashInfer bmm_fp8 on cuBLASLt: the heuristic's algorithm, what
   FlashInfer's autotune picks, and every algorithm cuBLASLt offers at 15
   rows), torch._scaled_mm, and FP8 Marlin W8A16 on the same FP8 bytes; GDN's
   BF16 in_proj_ba on cuBLAS against gb10_skinny's Triton GEMM and the FP8
   Marlin of FP8_SIDE=1; the autotune repeated three times (does it pick the
   same kernel?); the totals per decode step against the read roof. Needs ~1 GB:
   an idle server can stay up, a busy one skews the times. On the 27B alone:
       BENCH_SECTIONS=5 ~/spark/venv-sglang-0.5.21/bin/python patches/bench_sglang_gb10_gpu.py
6. the server's own FlashInfer autotune cache (~/.cache/sglang/flashinfer/autotune):
   for every cached FP8 tactic of those shapes, the cuBLASLt algorithm it becomes
   at 15 / 60 / 120 / 240 rows (a tactic is an index into the list cuBLASLt
   returns for the tuned bucket, 16 / 64 / 128 / 256 rows, taken at run time from
   the list for the real row count) against the heuristic, every algorithm at 15
   rows with its kernel, and bmm_fp8 with the file loaded as the server loads it.
       BENCH_SECTIONS=6 ~/spark/venv-sglang-0.5.21/bin/python patches/bench_sglang_gb10_gpu.py
7. the FP8 tactics tuned as the server tunes them (one pass at 240 and 256 rows,
   then the verify sizes run on what it stored), three times from scratch, with
   FlashInfer's buckets and with patches/gb10_fp8_tune.py (exact verify sizes):
   whether a size gets the same kernel every time, and the worst case per step.
       BENCH_SECTIONS=7 ~/spark/venv-sglang-0.5.21/bin/python patches/bench_sglang_gb10_gpu.py
8. the 27B's GEMMs at a 2048-row prefill chunk, which run FlashInfer's default
   tactic (SGLang skips its extend-size autotune with speculative decoding): the
   NVFP4 MLPs (gate_up, down) on every CUTLASS tactic, autotuned, and on the b12x
   and cudnn backends (also at 15 / 240 rows), the FP8 projections on every
   cuBLASLt algorithm; what the best would save a chunk.
       BENCH_SECTIONS=8 ~/spark/venv-sglang-0.5.21/bin/python patches/bench_sglang_gb10_gpu.py
Run straight from the venv's python, the script sets up what serve.sh would: the
venv's bin/ and CUDA's on PATH (FlashInfer's JIT runs a bare `ninja` and nvcc),
MAX_JOBS=2, and SGLang's cache directories, so FlashInfer finds the kernels the
server already compiled (~/.cache/sglang/.cache/flashinfer) instead of building
them again.
Exit code: the number of FAIL lines. BENCH_DRY=1: sections 1-3 on the CPU with
Triton's interpreter and tiny sizes (a check of this script; times mean nothing).
BENCH_SECTIONS=1,3: only those sections.
"""
import functools
import os
import sys
import time

DRY = os.environ.get("BENCH_DRY") == "1"
if DRY:
    os.environ["TRITON_INTERPRET"] = "1"
else:
    # What scripts/serve-sglang.sh and SGLang's redirect_third_party_caches() set,
    # before anything imports FlashInfer (it reads its workspace at import time).
    _paths = [os.path.dirname(sys.executable)]
    _cuda = os.environ.get("CUDA_HOME") or ("/usr/local/cuda" if os.path.exists("/usr/local/cuda/bin/nvcc") else "")
    if _cuda:
        os.environ.setdefault("CUDA_HOME", _cuda)
        _paths.append(os.path.join(_cuda, "bin"))
    os.environ["PATH"] = os.pathsep.join(_paths + [os.environ.get("PATH", "")])
    os.environ.setdefault("MAX_JOBS", "2")
    _base = os.path.expanduser(os.environ.get("SGLANG_CACHE_DIR", "~/.cache/sglang"))
    for _k, _v in (("FLASHINFER_WORKSPACE_BASE", _base), ("TRITON_CACHE_DIR", os.path.join(_base, "triton")),
                   ("CUDA_CACHE_PATH", os.path.join(_base, "nv"))):
        os.environ.setdefault(_k, _v)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gb10_fp8_hc as hcp  # noqa: E402
import gb10_fp8_hc_kernel as hck  # noqa: E402
from gb10_fp8_side import quantize_per_channel  # noqa: E402

SECTIONS = {int(s) for s in os.environ.get("BENCH_SECTIONS", "1,2,3,4,5,6,7,8").split(",") if s}
dev = torch.device("cpu" if DRY else "cuda")
DT = torch.float16 if DRY else torch.bfloat16
torch.manual_seed(0)
SMS = 1 if DRY else torch.cuda.get_device_properties(0).multi_processor_count
HC, HS, LOWRANK = (4, 64, 40) if DRY else (4, 2560, 320)
K = HC * HS
COPY_BYTES = 2e6 if DRY else 300e6
MIXES = 98   # a decode step at 1 stream: 2 x 48 layers, the final mixer, the MTP layer
fails = 0
print(f"{'CPU dry run' if DRY else torch.cuda.get_device_name(0)}, {SMS} SMs; torch {torch.__version__}")


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def graph_us(fn, calls, reps=5):
    """Time of one call of fn(i), i in range(calls), replayed from a CUDA graph."""
    def body():
        for i in range(calls):
            fn(i)
    if DRY:
        t = time.perf_counter()
        body()
        return (time.perf_counter() - t) * 1e6 / calls
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        body()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        body()
    g.replay()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(reps):
        g.replay()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) * 1000 / (calls * reps)


def eager_us(fn, reps=10, warm=2):
    if DRY:
        t = time.perf_counter()
        fn()
        return (time.perf_counter() - t) * 1e6
    for _ in range(warm):
        fn()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(reps):
        fn()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) * 1000 / reps


def mix_fp32(x, wd, sd, wu, su):
    wdf, wuf = wd.float() * sd[:, None], wu.float() * su[:, None]
    t = F.silu(x.float() @ wdf.T / HC)
    gate = torch.sigmoid(t @ wuf.T).unflatten(-1, (HC, HS))
    return (gate * x.float().unflatten(-1, (HC, HS))).mean(dim=-2)


def mix_eager_old(x, wd, sd, wu, su, hc, hs):
    """FP8_HC's prefill path until now: an eager chain over the [rows, hc*hs] gate."""
    t = F.linear(x, wd.to(x.dtype))
    t.mul_(sd.to(t.dtype)).div_(hc)
    t = F.silu(t)
    gate = F.linear(t, wu.to(x.dtype))
    gate.mul_(su.to(gate.dtype))
    torch.sigmoid_(gate)
    return (gate.unflatten(-1, (hc, hs)) * x.unflatten(-1, (hc, hs))).mean(dim=-2)


def mix_compute_bf16(x, wd, wu, hc, hs):
    """SGLang's _mix_compute (layers/hyperconnection.py), torch.compile'd there."""
    t = F.silu(F.linear(x, wd) / hc)
    gate = torch.sigmoid(F.linear(t, wu)).unflatten(-1, (hc, hs))
    return (gate * x.unflatten(-1, (hc, hs))).mean(dim=-2)


def make_mix():
    wd_b = (torch.randn(LOWRANK, K, device=dev) * 0.02).to(DT)
    wu_b = (torch.randn(K, LOWRANK, device=dev) * 0.05).to(DT)
    wd, sd = quantize_per_channel(wd_b)
    wu, su = quantize_per_channel(wu_b)
    return wd_b, wu_b, wd, sd.contiguous(), wu, su.contiguous()


def stock_kernel():
    if DRY:
        return None
    for mod in ("sglang.kernels.ops.gemm.hc_mix", "sglang.srt.layers.hc_mix_triton"):
        try:
            m = __import__(mod, fromlist=["fused_hc_mix"])
            return m.fused_hc_mix
        except Exception:  # noqa: BLE001 - not in this SGLang
            continue
    return None


copies = None
if SECTIONS & {1, 2}:
    n_copies = max(4, int(COPY_BYTES / (2 * LOWRANK * K)))
    copies = [make_mix() for _ in range(n_copies)]
    fp8_bytes = 2 * LOWRANK * K + 4 * (LOWRANK + K)

if 1 in SECTIONS:
    print(f"\n1. hyper-connection mix, decode sizes, from memory ({len(copies)} weight copies, CUDA graph)")
    stock = stock_kernel()
    for rows in ((1, 4, 17) if DRY else (1, 4, 8, 16, 32)):
        x = torch.randn(rows, K, device=dev).to(DT)
        _, _, wd, sd, wu, su = copies[0]
        out = hck.fused_hc_mix_fp8(x, wd, sd, wu, su, HC, HS, num_ctas=SMS)
        err = rel(out, mix_fp32(x, wd, sd, wu, su))
        # not a check: the kernel (like SGLang's BF16 one) sums its split-K
        # partials with atomic adds, so their order, and the last bits, vary
        same = torch.equal(out, hck.fused_hc_mix_fp8(x, wd, sd, wu, su, HC, HS, num_ctas=SMS))
        ok = err < 1e-2
        fails += not ok
        t8 = graph_us(lambda i: hck.fused_hc_mix_fp8(x, *copies[i][2:], HC, HS, num_ctas=SMS),
                      len(copies))
        line = (f"  {'PASS' if ok else 'FAIL'} {rows:3d} rows: FP8 {t8:6.1f} us "
                f"({fp8_bytes / t8 / 1e3:4.0f} GB/s), rel {err:.1e}, same bits twice: {same}")
        if stock is not None and rows <= 16:
            n_b = max(4, len(copies) // 2)
            tb = graph_us(lambda i: stock(x, copies[i][0], copies[i][1], HC, HS), n_b)
            line += f"; SGLang BF16 {tb:6.1f} us ({2 * fp8_bytes / tb / 1e3:4.0f} GB/s)"
        print(line)

if 2 in SECTIONS:
    print("\n2. FP8 mix tiles (BLOCK_N, BLOCK_K, BLOCK_J, BLOCK_R, warps, stages), from memory")
    for rows in ((4,) if DRY else (4, 32)):
        x = torch.randn(rows, K, device=dev).to(DT)
        rows_pad = 16 if rows <= 16 else 32 if rows <= 32 else 64
        default = hck.default_tiles(rows_pad)
        ref = mix_fp32(x, *copies[0][2:])
        if DRY:
            grid = [default, (32, 128, 8, 32, 4, 2)]
        else:
            bks = (128, 256) if rows_pad == 16 else (64, 128)
            stages = (2, 3) if rows_pad == 16 else (2,)
            grid = [default] + [(bn, bk, bj, 64, w, st) for bn in (16, 32) for bk in bks
                                for bj in (8, 16, 32) for w in (4, 8) for st in stages]
        results, seen = [], set()
        for cfg in grid:
            if cfg in seen:
                continue
            seen.add(cfg)
            try:
                out = hck.fused_hc_mix_fp8(x, *copies[0][2:], HC, HS, num_ctas=SMS, tiles=cfg)
                if rel(out, ref) > 1e-2:
                    print(f"  {rows} rows {cfg}: wrong result (rel {rel(out, ref):.1e}), skipped")
                    continue
                t = graph_us(lambda i: hck.fused_hc_mix_fp8(x, *copies[i][2:], HC, HS,
                                                            num_ctas=SMS, tiles=cfg),
                             len(copies), reps=3)
                results.append((cfg, t))
            except Exception as e:  # noqa: BLE001 - a tile this GPU cannot build
                print(f"  {rows} rows {cfg}: {type(e).__name__}: {str(e).splitlines()[0][:100]}")
        if not results:
            print(f"  {rows} rows: no tile ran")
            fails += 1
            continue
        d = next((t for c, t in results if c == default), None)
        results.sort(key=lambda r: r[1])
        best = "; ".join(f"{c} {t:.1f} us" for c, t in results[:5])
        print(f"  {rows:2d} rows: default {default} {d:.1f} us" if d is not None
              else f"  {rows:2d} rows: default {default} did not run")
        print(f"           best {best}")

if 3 in SECTIONS:
    print("\n3. hyper-connection mix, prefill sizes (eager, one mix)")
    wd_b, wu_b, wd, sd, wu, su = make_mix()
    compiled = None
    if not DRY:
        try:
            compiled = torch.compile(mix_compute_bf16)
        except Exception as e:  # noqa: BLE001
            print(f"  torch.compile unavailable: {type(e).__name__}")
    for rows in ((96,) if DRY else (2048, 4096, 8192)):
        x = torch.randn(rows, K, device=dev).to(DT)
        new = hcp.mix_reference(x, wd, sd, wu, su, HC, HS)
        old = mix_eager_old(x, wd, sd, wu, su, HC, HS)
        err = rel(new, mix_fp32(x, wd, sd, wu, su))
        ok = err < 1e-2 and hcp._prefill["ok"]
        fails += not ok
        t_old = eager_us(lambda: mix_eager_old(x, wd, sd, wu, su, HC, HS))
        t_new = eager_us(lambda: hcp.mix_reference(x, wd, sd, wu, su, HC, HS))
        line = (f"  {'PASS' if ok else 'FAIL'} {rows:5d} rows: FP8 eager chain {t_old:8.1f} us, "
                f"FP8 new {t_new:8.1f} us (x{t_old / t_new:.2f}; rel {err:.1e}, "
                f"old-new {rel(new, old):.1e})")
        if compiled is not None and rows == 4096:
            try:
                tc = eager_us(lambda: compiled(x, wd_b, wu_b, HC, HS), warm=3)
                line += f"; SGLang BF16 compiled {tc:8.1f} us"
            except Exception as e:  # noqa: BLE001
                line += f"; SGLang BF16 compiled failed ({type(e).__name__})"
        print(line)
        if rows == 4096:
            print(f"        a 4096-token chunk has {MIXES - 1} mixes: {(t_old - t_new) * (MIXES - 1) / 1e3:+.1f} ms "
                  "saved per chunk (positive = faster)")
        del x, new, old

if 4 in SECTIONS and not DRY:
    print("\n4. FP8 side layers at prefill sizes: time per call (eager), us")
    import gb10_fp8_side as side
    from sglang.srt.layers.quantization.marlin_utils_fp8 import (
        apply_fp8_marlin_linear,
        prepare_fp8_layer_for_marlin,
    )

    w8a8 = {}
    try:
        from sglang.kernels.ops.gemm import fp8_scaled_mm
        from sglang.kernels.ops.quantization.fp8_kernel import (
            sglang_per_token_quant_fp8,
        )

        def _sgl(x, q, s):
            xq, xs = sglang_per_token_quant_fp8(x)
            return fp8_scaled_mm(xq, q.t(), xs, s.view(-1, 1), x.dtype)
        w8a8["sgl"] = _sgl
    except Exception as e:  # noqa: BLE001
        print(f"  SGLang fp8_scaled_mm unavailable: {type(e).__name__}: {e}")
    try:
        from sglang.kernels.ops.gemm.fp8_kernel import triton_scaled_mm
        from sglang.kernels.ops.quantization.fp8_kernel import (
            sglang_per_token_quant_fp8 as _ptq,
        )

        def _tri(x, q, s):
            xq, xs = _ptq(x)
            return triton_scaled_mm(xq, q.t(), xs, s.view(-1, 1), x.dtype)
        w8a8["triton"] = _tri
    except Exception as e:  # noqa: BLE001
        print(f"  SGLang triton_scaled_mm unavailable: {type(e).__name__}: {e}")

    def _quant_rows(x):
        amax = x.abs().amax(dim=1, keepdim=True).float().clamp_min(1e-12)
        s = amax / 448.0
        return (x.float() / s).clamp(-448, 448).to(torch.float8_e4m3fn), s

    def _torch_scaled(x, q, s):
        xq, xs = _quant_rows(x)
        return torch._scaled_mm(xq, q.t(), scale_a=xs, scale_b=s.view(1, -1).float(),
                                out_dtype=x.dtype)
    w8a8["torch"] = _torch_scaled

    import gb10_fp8_prefill as pre

    unpack_ok = pre.setup(prepare_fp8_layer_for_marlin, dev)
    print(f"  gb10_fp8_prefill.setup: {'ready' if unpack_ok else 'disabled: ' + pre._state['why']}")
    fails += not unpack_ok

    # (name, N, K, layers in the model)
    shapes = [("GDN in_proj_qkvz", 16384, 2560, 36), ("GDN in_proj_ba (padded)", 128, 2560, 36),
              ("GDN out_proj", 2560, 6144, 36), ("attn qkv_proj", 13312, 2560, 12),
              ("attn o_proj", 2560, 6144, 12), ("shared gate_up", 1280, 2560, 48),
              ("shared down", 2560, 640, 48)]
    names = ["marlin", "bf16", "dq+bf16"] + [f"w8a8-{k}" for k in w8a8] + ["marlin->w8a8"]
    chunk = {n: 0.0 for n in names}
    for label, n, k, count in shapes:
        w = (torch.randn(n, k, device=dev) * 0.02).to(torch.bfloat16)
        q, s = quantize_per_channel(w)
        wdq = (q.float() * s[:, None]).to(torch.bfloat16)
        m = torch.nn.Module()
        m.weight = torch.nn.Parameter(w.clone(), requires_grad=False)
        side.convert_module(m, prepare_fp8_layer_for_marlin, apply_fp8_marlin_linear)
        fns = {
            "marlin": lambda x: m.quant_method.apply(m, x),
            "bf16": lambda x: F.linear(x, wdq),
            "dq+bf16": lambda x: F.linear(x, q.to(torch.bfloat16).mul_(s[:, None].to(torch.bfloat16))),
        }
        for key, fn in w8a8.items():
            fns[f"w8a8-{key}"] = functools.partial(lambda f, x: f(x, q, s), fn)
        note = ""
        if unpack_ok:
            # the server's path (FP8_SIDE_PREFILL=w8a8): restore the plain weight from
            # Marlin's layout, then fp8_scaled_mm (m is not marked, so "marlin" stays Marlin)
            m._gb10_wscale_col = m._gb10_wscale.float().view(-1, 1).contiguous()
            plain = pre.unpack(m.weight, m._gb10_marlin_n, k, pre._state["inv"])
            exact = torch.equal(plain.view(torch.uint8), q.view(torch.uint8))
            fails += not exact
            t_un = eager_us(lambda: pre.unpack(m.weight, m._gb10_marlin_n, k, pre._state["inv"]))
            note = (f" (weight restored from Marlin's layout {'bit for bit' if exact else 'WRONG: FAIL'},"
                    f" {t_un:.0f} us a call)")
            fns["marlin->w8a8"] = lambda x: pre.apply_w8a8(m, x)
        print(f"  {label} [{n} x {k}] x{count}:{note}")
        for rows in (4, 512, 2048, 4096, 8192):
            x = torch.randn(rows, k, device=dev).to(torch.bfloat16)
            ref = x.float() @ wdq.float().T
            cells = []
            for name, fn in fns.items():
                try:
                    out = fn(x)
                    e = rel(out, ref)
                    if name == "marlin" and e > 1e-2:
                        fails += 1
                        cells.append(f"{name} FAIL rel {e:.1e}")
                        continue
                    t = eager_us(lambda: fn(x))
                    tf = 2 * rows * n * k / t / 1e6
                    cells.append(f"{name} {t:8.1f} ({tf:5.1f} TF, rel {e:.0e})")
                    if rows == 4096:
                        chunk[name] += t * count
                except Exception as ex:  # noqa: BLE001
                    cells.append(f"{name} n/a ({type(ex).__name__})")
                    if rows == 4096:
                        chunk[name] = float("nan")
            print(f"    {rows:5d} rows: " + " | ".join(cells))
            del x
        del m, w, q, wdq
        torch.cuda.empty_cache()
    print("  a 4096-token chunk, all side layers: "
          + ", ".join(f"{k} {v / 1e3:.0f} ms" for k, v in chunk.items() if v != 0))

if SECTIONS & {5, 6, 7, 8} and not DRY:
    print("\n5-8. Qwen3.8-27B (models/qwen3.8-27b.sh) GEMMs")
    import flashinfer
    import gb10_fp8_side as side
    from flashinfer.autotuner import AutoTuner, autotune
    from flashinfer.gemm import gemm_base as fgb
    from gb10_skinny_kernel import skinny_gemm
    from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8
    from sglang.srt.layers.quantization.marlin_utils_fp8 import (
        apply_fp8_marlin_linear,
        prepare_fp8_layer_for_marlin,
    )
    from torch.profiler import ProfilerActivity, profile

    FP8 = torch.float8_e4m3fn
    ROWS27 = (15, 60, 120, 240)
    L2 = torch.cuda.get_device_properties(0).L2_cache_size

    def kernel_names(fn):
        """The GPU kernels one call of fn launches (names cut to 75 characters)."""
        fn()
        torch.cuda.synchronize()
        try:
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                fn()
                torch.cuda.synchronize()
        except Exception as ex:  # noqa: BLE001  (no CUPTI: times still count)
            return f"(kernel names n/a: {type(ex).__name__})"
        names = []
        for e in prof.events():
            if e.device_type == torch.autograd.DeviceType.CUDA and not e.name.startswith(("Memcpy", "Memset")):
                if e.name[:75] not in names:
                    names.append(e.name[:75])
        return " + ".join(names) or "?"

    def copies_for(nbytes):
        return max(2, -(-int(max(COPY_BYTES, 3 * L2)) // nbytes))

    big = torch.zeros(1 << 28, dtype=torch.float32, device=dev)   # 1 GiB
    dst = torch.empty_like(big)
    t_sum = eager_us(lambda: big.sum(), reps=5)
    t_max = eager_us(lambda: big.view(torch.bfloat16).amax(), reps=5)
    t_cp = eager_us(lambda: dst.copy_(big), reps=5)
    roof = max((1 << 30) / t_sum, (1 << 30) / t_max) / 1e3
    del big, dst
    print(f"  read roof over 1 GiB: {(1 << 30) / t_sum / 1e3:.0f} GB/s (fp32 sum), "
          f"{(1 << 30) / t_max / 1e3:.0f} GB/s (bf16 amax); copy {2 * (1 << 30) / t_cp / 1e3:.0f} GB/s "
          f"(read + write); L2 {L2 / 2**20:.0f} MiB. Percentages below are of {roof:.0f} GB/s.")

    def gbs(nbytes, us):
        return nbytes / us / 1e3

    runner = fgb.get_gemm_module().cublas_fp8_gemm_runner()
    wsbuf = fgb._get_cache_buf("bmm_fp8_workspace", fgb.DEFAULT_WORKSPACE_SIZE, dev)
    FP8_SHAPES_27B = [("GDN out_proj + attn o_proj", 5120, 6144, 64),
                      ("GDN in_proj_qkvz", 16384, 5120, 48),
                      ("attn qkv_proj", 14336, 5120, 16)]

    def fp8_weights(n, k, copies):
        """FP8 per-tensor weight copies [N, K], the [1, K, N] column-major views bmm_fp8
        takes, the per-tensor scale and the dequantized reference."""
        w = (torch.randn(n, k, device=dev) * 0.02).to(torch.bfloat16)
        wscale = (w.abs().amax().float() / 448.0).reshape(1)
        q = (w.float() / wscale).clamp(-448, 448).to(FP8)
        qs = [q.clone() for _ in range(copies)]
        return qs, [c.t().unsqueeze(0) for c in qs], wscale, q.float() * wscale

    def fp8_input(rows, k):
        x = torch.randn(rows, k, device=dev).to(torch.bfloat16)
        xs = (x.abs().amax().float() / 448.0).reshape(1)
        return x, xs, static_quant_fp8(x, xs)[0]

if 5 in SECTIONS and not DRY:
    print("\n5. Qwen3.8-27B decode GEMMs at verify sizes (DFlash2 draft 15: 15 / 60 / 120 / 240 rows ="
          " 1 / 4 / 8 / 16 streams), us a call; weights cycled through copies, so they come from memory")
    per_step = {}   # (layer label, variant) -> ms per decode step at 15 rows
    for label, n, k, count in FP8_SHAPES_27B:
        nbytes = n * k
        copies = copies_for(nbytes)
        w = (torch.randn(n, k, device=dev) * 0.02).to(torch.bfloat16)
        wscale = (w.abs().amax().float() / 448.0).reshape(1)
        q = (w.float() / wscale).clamp(-448, 448).to(FP8)
        wref = q.float() * wscale
        del w
        qs = [q.clone() for _ in range(copies)]
        wt = [c.t().unsqueeze(0) for c in qs]   # [1, K, N] column-major, as SGLang hands bmm_fp8 its weight
        marl = []
        for c in qs:
            mod = torch.nn.Module()
            mod.weight = torch.nn.Parameter(c, requires_grad=False)
            mod.weight_scale = torch.nn.Parameter(wscale.clone(), requires_grad=False)
            mod.orig_dtype = torch.bfloat16
            mod.output_size_per_partition, mod.input_size_per_partition = n, k
            prepare_fp8_layer_for_marlin(mod, size_k_first=False)
            marl.append(mod)
        print(f"  {label} [{n} x {k}] FP8 per-tensor x{count} layers ({nbytes / 1e6:.1f} MB each, {copies} copies):")
        for rows in ROWS27:
            x = torch.randn(rows, k, device=dev).to(torch.bfloat16)
            xs = (x.abs().amax().float() / 448.0).reshape(1)
            xq = static_quant_fp8(x, xs)[0]
            xq3 = xq.unsqueeze(0)
            out3 = torch.empty(1, rows, n, device=dev, dtype=torch.bfloat16)
            ref = x.float() @ wref.T

            def bmm(i, xq3=xq3, out3=out3, xs=xs):
                return flashinfer.bmm_fp8(xq3, wt[i % copies], xs, wscale, torch.bfloat16,
                                          out=out3, backend="cublas")

            def marlin(i, x=x):
                m = marl[i % copies]
                return apply_fp8_marlin_linear(input=x, weight=m.weight, weight_scale=m.weight_scale,
                                               workspace=m.workspace, size_n=n, size_k=k, bias=None)

            def torch_mm(i, xq=xq, xs=xs):
                return torch._scaled_mm(xq, qs[i % copies].t(), scale_a=xs, scale_b=wscale,
                                        out_dtype=torch.bfloat16)

            calls = copies * 3
            t_quant = graph_us(lambda i: static_quant_fp8(x, xs), 20)
            cells = {}
            AutoTuner.get().clear_cache()
            cells["cuBLASLt heuristic"] = (graph_us(bmm, calls), rel(bmm(0), ref), kernel_names(lambda: bmm(0)))
            picks = []
            for _ in range(3 if rows == 15 else 1):
                AutoTuner.get().clear_cache()
                with autotune(True):
                    bmm(0)
                torch.cuda.synchronize()
                picks.append(kernel_names(lambda: bmm(0)))
            cells["cuBLASLt autotuned"] = (graph_us(bmm, calls), rel(bmm(0), ref), picks[-1])
            try:
                cells["torch._scaled_mm"] = (graph_us(torch_mm, calls), rel(torch_mm(0), ref),
                                             kernel_names(lambda: torch_mm(0)))
            except Exception as ex:  # noqa: BLE001
                print(f"    torch._scaled_mm n/a: {type(ex).__name__}: {str(ex)[:120]}")
            out = marlin(0)
            e = rel(out, ref)
            if e > 3e-2:
                fails += 1
                print(f"    marlin FAIL: rel {e:.1e}")
            cells["Marlin W8A16"] = (graph_us(marlin, calls), e, kernel_names(lambda: marlin(0)))
            sweep = []
            if rows == 15:
                def inputs(i, xq3=xq3, out3=out3, xs=xs):
                    return [xq3, wt[i % copies], xs, wscale, out3, wsbuf]
                for tac in runner.get_valid_tactics(inputs(0), None):
                    try:
                        t = graph_us(lambda i, tac=tac: runner(inputs(i), tactic=tac), calls)
                    except Exception:  # noqa: BLE001
                        continue
                    sweep.append((t, tac))
                sweep.sort()
                if sweep:
                    t, tac = sweep[0]
                    runner(inputs(0), tactic=tac)
                    cells[f"cuBLASLt best of {len(sweep)} (#{tac})"] = (
                        t, rel(out3[0], ref), kernel_names(lambda: runner(inputs(0), tactic=tac)))
            print(f"    {rows:3d} rows (activation quant {t_quant:.1f} us extra for the cuBLASLt rows):")
            for name, (t, e, kern) in cells.items():
                print(f"      {name:28s} {t:7.1f} us {gbs(nbytes, t):5.0f} GB/s ({100 * gbs(nbytes, t) / roof:3.0f}%)"
                      f"  rel {e:.0e}  {kern}")
                if rows == 15:
                    per_step[(label, name.split(" (#")[0].split(" of ")[0])] = t * count / 1e3
            if rows == 15:
                same = len(set(picks)) == 1
                print(f"      autotune, {len(picks)} times from scratch: "
                      + ("the same kernel each time" if same else "DIFFERENT kernels: " + " | ".join(picks)))
                if sweep:
                    print("      cuBLASLt algorithms at 15 rows, fastest first: "
                          + ", ".join(f"#{tac} {t:.0f}" for t, tac in sweep[:12])
                          + (f" ... slowest #{sweep[-1][1]} {sweep[-1][0]:.0f} us" if len(sweep) > 12 else ""))
            del x, xq, xq3, out3, ref
        del qs, wt, marl, q, wref
        torch.cuda.empty_cache()

    # GDN in_proj_ba: BF16 in the checkpoint, 96 outputs
    n, k, count = 96, 5120, 48
    nbytes = n * k * 2
    copies = copies_for(nbytes)
    wbf = [(torch.randn(n, k, device=dev) * 0.02).to(torch.bfloat16) for _ in range(copies)]
    mods = []
    for c in wbf:
        m = torch.nn.Module()
        m.weight = torch.nn.Parameter(c.clone(), requires_grad=False)
        side.convert_module(m, prepare_fp8_layer_for_marlin, apply_fp8_marlin_linear)
        mods.append(m)
    print(f"  GDN in_proj_ba [{n} x {k}] BF16 x{count} layers ({nbytes / 1e6:.1f} MB each, {copies} copies):")
    for rows in ROWS27:
        x = torch.randn(rows, k, device=dev).to(torch.bfloat16)
        ref = x.float() @ wbf[0].float().T
        fns = {"cuBLAS (stock)": lambda i, x=x: F.linear(x, wbf[i % copies]),
               "skinny Triton (BF16)": lambda i, x=x: skinny_gemm(x, wbf[i % copies]),
               "FP8 Marlin (FP8_SIDE=1)": lambda i, x=x: mods[i % copies].quant_method.apply(mods[i % copies], x)}
        print(f"    {rows:3d} rows:")
        for name, fn in fns.items():
            try:
                e = rel(fn(0), ref)
                if name.startswith("skinny") and e > 1e-2:
                    fails += 1
                    print(f"      {name} FAIL: rel {e:.1e}")
                    continue
                t = graph_us(fn, copies)
                print(f"      {name:28s} {t:7.1f} us {gbs(nbytes, t):5.0f} GB/s  rel {e:.0e}  "
                      f"{kernel_names(lambda: fn(0))}")
                if rows == 15:
                    per_step[("GDN in_proj_ba", name)] = t * count / 1e3
            except Exception as ex:  # noqa: BLE001
                print(f"      {name} n/a ({type(ex).__name__}: {str(ex)[:100]})")
        del x
    del wbf, mods
    torch.cuda.empty_cache()

    print("  per decode step at 1 stream (15 rows), ms; the decode profile of 2026-11-10 had ~104 ms a step:")
    for (label, name), ms in per_step.items():
        print(f"    {label:28s} {name:28s} {ms:6.2f}")

if 6 in SECTIONS and not DRY:
    import glob
    import json
    import re

    print("\n6. The server's own FlashInfer autotune cache: what a cached FP8 tactic runs at the verify"
          " sizes. A tactic is an index into the list cuBLASLt returns for the tuned shape (bucket M:"
          " 16 / 64 / 128 / 256); at run time the same index is taken from the list for the real M"
          " (15 / 60 / 120 / 240)")
    tune_dir = os.path.join(os.environ["FLASHINFER_WORKSPACE_BASE"], "flashinfer", "autotune")
    files = sorted(glob.glob(os.path.join(tune_dir, "**", "*.json"), recursive=True), key=os.path.getmtime)
    key_rx = re.compile(r"'fp8_gemm', '(\w+)', \(\(1, (\d+), (\d+)\), \(1, (\d+), (\d+)\)")
    cached = []   # (file, {(k, n): {bucket: tactic}})

    def tag(f):   # SGLang's cache key directory (a hash of model path, quantization, ...)
        return os.path.basename(os.path.dirname(f))[:16]
    for f in files:
        try:
            data = json.load(open(f))
        except (OSError, ValueError) as ex:
            print(f"  {f}: unreadable ({type(ex).__name__})")
            continue
        per_shape = {}
        n_fp8 = n_all = 0
        for key, val in data.items():
            if key.startswith("_"):
                continue
            n_all += 1
            m = key_rx.search(key)
            if m and isinstance(val, list) and len(val) == 2 and isinstance(val[1], int):
                n_fp8 += 1
                per_shape.setdefault((int(m[3]), int(m[5])), {})[int(m[2])] = val[1]
        stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(f)))
        print(f"  {os.path.relpath(f, tune_dir)} ({stamp}): {n_all} entries, {n_fp8} fp8_gemm"
              + (" for " + ", ".join(f"[K {k} x N {n}]" for k, n in sorted(per_shape)) if per_shape else ""))
        if any((k, n) in per_shape for _, n, k, _ in FP8_SHAPES_27B):
            cached.append((f, per_shape))
    if not files:
        print(f"  no cache files under {tune_dir}")
    elif not cached:
        print("  none of them has the 27B's FP8 shapes")
    bucket_of = {15: 16, 60: 64, 120: 128, 240: 256}
    for label, n, k, count in FP8_SHAPES_27B:
        if not any((k, n) in ps for _, ps in cached):
            continue
        copies = copies_for(n * k)
        qs, wt, wscale, _ = fp8_weights(n, k, copies)
        print(f"  {label} [{n} x {k}] x{count}:")
        for rows in ROWS27:
            outs = {}
            x, xs, xq = fp8_input(rows, k)
            for m_ in sorted({rows, bucket_of[rows]}):
                xq_m = xq if m_ == rows else fp8_input(m_, k)[2]
                outs[m_] = (xq_m.unsqueeze(0), torch.empty(1, m_, n, device=dev, dtype=torch.bfloat16))

            def run(i, tac, m_):
                a3, o3 = outs[m_]
                return runner([a3, wt[i % copies], xs, wscale, o3, wsbuf], tactic=tac)

            def timed(tac, m_):
                return graph_us(lambda i: run(i, tac, m_), copies * 3), kernel_names(lambda: run(0, tac, m_))

            t_h, k_h = timed(-1, rows)
            count_r = len(runner.get_valid_tactics([outs[rows][0], wt[0], xs, wscale, outs[rows][1], wsbuf], None))
            print(f"    {rows:3d} rows: cuBLASLt heuristic {t_h:6.1f} us  {k_h}  ({count_r} algorithms at {rows} rows)")
            if rows == 15:
                lst = []
                for tac in range(count_r):
                    try:
                        lst.append((tac, *timed(tac, rows)))
                    except Exception:  # noqa: BLE001
                        lst.append((tac, float("nan"), "failed"))
                for tac, t, kern in lst:
                    print(f"           #{tac}: {t:6.1f} us  {kern}")
            for f, ps in cached:
                tac = ps.get((k, n), {}).get(bucket_of[rows])
                if tac is None:
                    print(f"      {tag(f)}: no entry for bucket {bucket_of[rows]}")
                    continue
                t_r, k_r = timed(tac, rows)
                t_b, k_b = timed(tac, bucket_of[rows])
                AutoTuner.get().clear_cache()
                loaded = AutoTuner.get().load_configs(f)
                a3, o3 = outs[rows]
                t_s = graph_us(lambda i, a3=a3, o3=o3: flashinfer.bmm_fp8(
                    a3, wt[i % copies], xs, wscale, torch.bfloat16, out=o3, backend="cublas"), copies * 3)
                k_s = kernel_names(lambda a3=a3, o3=o3: flashinfer.bmm_fp8(
                    a3, wt[0], xs, wscale, torch.bfloat16, out=o3, backend="cublas"))
                AutoTuner.get().clear_cache()
                print(f"      {tag(f)}: cached tactic #{tac} (bucket {bucket_of[rows]})"
                      f" -> at {rows} rows {t_r:6.1f} us {k_r}; at {bucket_of[rows]} rows {t_b:6.1f} us {k_b}")
                print(f"        bmm_fp8 with this file loaded{'' if loaded else ' (FlashInfer refused the file)'}:"
                      f" {t_s:6.1f} us {k_s}; against the heuristic {(t_s - t_h) * count / 1e3:+.2f} ms a step")
            del x, xs, xq, outs
        del qs, wt
        torch.cuda.empty_cache()

if 7 in SECTIONS and not DRY:
    import gb10_fp8_tune as fp8_tune

    print("\n7. FP8 tactics tuned as the server tunes them (one pass at 240 and 256 rows inside"
          " autotune, then the verify sizes run on whatever it stored), three times from scratch:"
          " FlashInfer's buckets (stock) against patches/gb10_fp8_tune.py (exact verify sizes,"
          " FP8_GEMM_TUNE=exact)")
    orig_cfg = getattr(fgb, fp8_tune.CONFIG_NAME)

    def tune_mode(mode):
        setattr(fgb, fp8_tune.CONFIG_NAME, orig_cfg)
        for attr in (fp8_tune._MARK, "_gb10_fp8_tune_orig"):
            if hasattr(fgb, attr):
                delattr(fgb, attr)
        if mode == "exact":
            os.environ["GB10_FP8_TUNE_ROWS"], os.environ["GB10_FP8_TUNE_MAX_BS"] = "15", "16"
            fp8_tune.apply(fgb)

    for label, n, k, count in FP8_SHAPES_27B:
        copies = copies_for(n * k)
        qs, wt, wscale, _ = fp8_weights(n, k, copies)
        ins = {r: fp8_input(r, k) for r in ROWS27 + (256,)}
        outs = {r: torch.empty(1, r, n, device=dev, dtype=torch.bfloat16) for r in ins}

        def bmm7(i, r):
            _, xs_r, xq_r = ins[r]
            return flashinfer.bmm_fp8(xq_r.unsqueeze(0), wt[i % copies], xs_r, wscale, torch.bfloat16,
                                      out=outs[r], backend="cublas")

        print(f"  {label} [{n} x {k}] x{count}:")
        step = {}
        for mode in ("stock", "exact"):
            tune_mode(mode)
            res = {r: [] for r in ROWS27}
            for _ in range(3):
                AutoTuner.get().clear_cache()
                with autotune(True):
                    bmm7(0, 240)
                    bmm7(0, 256)
                torch.cuda.synchronize()
                for r in ROWS27:
                    res[r].append((graph_us(lambda i, r=r: bmm7(i, r), copies * 3), kernel_names(lambda r=r: bmm7(0, r))))
            for r in ROWS27:
                kerns = sorted({kn for _, kn in res[r]})
                print(f"    {r:3d} rows, {mode:5s}: " + " / ".join(f"{t:6.1f}" for t, _ in res[r]) + " us  "
                      + (kerns[0] if len(kerns) == 1 else f"{len(kerns)} different kernels: " + " | ".join(kerns)))
                step[(mode, r)] = max(t for t, _ in res[r]) * count / 1e3
        AutoTuner.get().clear_cache()
        tune_mode("stock")
        print("    worst of three, ms a verify step for these layers: " + ", ".join(
            f"{r} rows {step[('stock', r)]:.2f} -> {step[('exact', r)]:.2f}" for r in ROWS27))
        del qs, wt, ins, outs
        torch.cuda.empty_cache()
    os.environ.pop("GB10_FP8_TUNE_ROWS", None)
    os.environ.pop("GB10_FP8_TUNE_MAX_BS", None)

if 8 in SECTIONS and not DRY:
    from flashinfer import SfLayout

    print("\n8. Prefill-size GEMMs of the 27B: with DFlash2 SGLang skips its extend-size autotune, so a"
          " 2048-row chunk runs FlashInfer's default tactic. Every tactic against that default"
          " (NVFP4 MLPs on CUTLASS, also the b12x / cudnn backends; FP8 projections on cuBLASLt), and"
          " at 15 / 240 rows for reference. GEMM time only: the activation quantization is the same"
          " for every tactic. \"autotuned\" finds its tuning only at 2048 rows: FlashInfer looks 15 / 240"
          " up at 16 / 256, so there it is the default")
    cap = torch.cuda.get_device_capability()
    fp4_runner = fgb.get_cutlass_fp4_gemm_module(*cap).cutlass_fp4_gemm_runner()
    fp4_ws = fgb._get_cache_buf("mm_fp4_workspace", fgb.DEFAULT_WORKSPACE_SIZE, dev)
    CHUNK_MS = 811   # GPU time of a 2048-token chunk at 2-16K context (profile_decode --prefill 16384)

    def nvfp4(t):
        gsf = ((448 * 6) / t.float().abs().amax().clamp_min(1e-12)).reshape(1)
        q, sf = flashinfer.nvfp4_quantize(t, gsf, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
        return q, sf, gsf

    saved = []   # (label, ms a 2048-row chunk, what)
    for label, n, k, count in (("MLP gate_up", 34816, 5120, 64), ("MLP down", 5120, 17408, 64)):
        copies = copies_for(n * k // 2 + n * k // 16)
        w0 = (torch.randn(n, k, device=dev) * 0.02).to(torch.bfloat16)
        wq = [nvfp4(w0)]
        for _ in range(copies - 1):
            wq.append(nvfp4((torch.randn(n, k, device=dev) * 0.02).to(torch.bfloat16)))
        print(f"  {label} [{n} x {k}] NVFP4 x{count} layers ({copies} copies), "
              f"{len(fp4_runner.get_valid_tactics([], None))} CUTLASS tactics:")
        for rows in (15, 240, 2048):
            x = torch.randn(rows, k, device=dev).to(torch.bfloat16)
            a, a_sf, a_gsf = nvfp4(x)
            alphas = [(1.0 / (a_gsf * g)).float() for _, _, g in wq]
            out = torch.empty(rows, n, device=dev, dtype=torch.bfloat16)
            ref = (x @ w0.T).float()
            tflop = 2 * rows * n * k / 1e12

            def mm(i, backend="cutlass", a=a, a_sf=a_sf, alphas=alphas, out=out):
                q, sf, _ = wq[i % copies]
                return flashinfer.mm_fp4(a, q.T, a_sf, sf.T, alphas[i % copies], torch.bfloat16, out,
                                         backend=backend)

            def tac(i, t, a=a, a_sf=a_sf, alphas=alphas, out=out):
                q, sf, _ = wq[i % copies]
                return fp4_runner([a, q.T, a_sf, sf.T, alphas[i % copies], torch.bfloat16, out, 16, True,
                                   fp4_ws], tactic=t)

            calls = copies * 2

            def line(name, t, e, kern=""):
                print(f"      {name:30s} {t:8.1f} us  {tflop / t * 1e6:6.1f} TFLOPS  rel {e:.0e}  {kern}")

            print(f"    {rows} rows:")
            AutoTuner.get().clear_cache()
            t_def = graph_us(mm, calls)
            ref_def = mm(0).float().clone()
            e = rel(ref_def, ref)
            if e > 0.3:
                fails += 1
                print(f"      FAIL: CUTLASS default rel {e:.1e} against BF16 (layout?)")
            line("CUTLASS default (the server)", t_def, e, kernel_names(lambda: mm(0)))
            best = ("CUTLASS default", t_def)
            if rows != 15:
                sweep = []
                for t in fp4_runner.get_valid_tactics([], None):
                    try:
                        tt = graph_us(lambda i, t=t: tac(i, t), calls)
                        et = rel(tac(0, t).float(), ref_def)
                    except Exception:  # noqa: BLE001
                        continue
                    if et > 2e-2:
                        print(f"      tactic #{t}: differs from the default by rel {et:.1e}, left out")
                        continue
                    sweep.append((tt, t))
                sweep.sort()
                if sweep:
                    tt, t = sweep[0]
                    line(f"CUTLASS best of {len(sweep)} (#{t})", tt, rel(tac(0, t).float(), ref),
                         kernel_names(lambda t=t: tac(0, t)))
                    print("      all tactics, fastest first: " + ", ".join(f"#{t} {tt:.0f}" for tt, t in sweep))
                    best = min(best, (f"CUTLASS tactic #{t}", tt), key=lambda b: b[1])
            AutoTuner.get().clear_cache()
            with autotune(True):
                mm(0)
            torch.cuda.synchronize()
            tt = graph_us(mm, calls)
            line("CUTLASS autotuned here", tt, rel(mm(0).float(), ref), kernel_names(lambda: mm(0)))
            for be in ("b12x", "cudnn"):
                for tuned in (False, True):
                    AutoTuner.get().clear_cache()
                    try:
                        if tuned:
                            with autotune(True):
                                mm(0, be)
                            torch.cuda.synchronize()
                        tt = graph_us(lambda i, be=be: mm(i, be), calls)
                        e = rel(mm(0, be).float(), ref)
                    except Exception as ex:  # noqa: BLE001
                        print(f"      {be:6s}{' autotuned' if tuned else ''}: n/a ({type(ex).__name__}: {str(ex)[:90]})")
                        break
                    line(f"{be}{' autotuned' if tuned else ' default'}", tt, e, kernel_names(lambda be=be: mm(0, be)))
                    if e < 0.3:
                        best = min(best, (f"{be}{' autotuned' if tuned else ''}", tt), key=lambda b: b[1])
            AutoTuner.get().clear_cache()
            if rows == 2048:
                saved.append((label, (t_def - best[1]) * count / 1e3, best[0]))
            del x, a, a_sf, out, ref, ref_def
        del wq, w0
        torch.cuda.empty_cache()

    for label, n, k, count in FP8_SHAPES_27B:
        copies = copies_for(n * k)
        qs, wt, wscale, wref = fp8_weights(n, k, copies)
        x, xs, xq = fp8_input(2048, k)
        a3 = xq.unsqueeze(0)
        o3 = torch.empty(1, 2048, n, device=dev, dtype=torch.bfloat16)
        ref = (x.float() @ wref.T)
        tflop = 2 * 2048 * n * k / 1e12

        def run8(i, t, a3=a3, o3=o3, xs=xs):
            return runner([a3, wt[i % copies], xs, wscale, o3, wsbuf], tactic=t)

        t_h = graph_us(lambda i: run8(i, -1), copies * 2)
        k_h = kernel_names(lambda: run8(0, -1))
        sweep = []
        for t in runner.get_valid_tactics([a3, wt[0], xs, wscale, o3, wsbuf], None):
            try:
                sweep.append((graph_us(lambda i, t=t: run8(i, t), copies * 2), t))
            except Exception:  # noqa: BLE001
                continue
        sweep.sort()
        tb, tb_i = sweep[0] if sweep else (t_h, -1)
        e = rel(run8(0, tb_i)[0].float(), ref)
        print(f"  {label} [{n} x {k}] FP8 x{count}, 2048 rows: cuBLASLt heuristic (the server) {t_h:7.1f} us"
              f" {tflop / t_h * 1e6:5.1f} TFLOPS {k_h}; best of {len(sweep)} #{tb_i} {tb:7.1f} us"
              f" {tflop / tb * 1e6:5.1f} TFLOPS rel {e:.0e} {kernel_names(lambda: run8(0, tb_i))}")
        saved.append((label, max(0.0, t_h - tb) * count / 1e3, f"cuBLASLt #{tb_i}"))
        del qs, wt, x, xq, a3, o3, ref, wref
        torch.cuda.empty_cache()

    total = sum(ms for _, ms, _ in saved)
    print(f"  a 2048-token chunk (~{CHUNK_MS} ms of GPU time at 2-16K context), best against what runs now:")
    for label, ms, what in saved:
        print(f"    {label:28s} {ms:6.1f} ms saved ({what})")
    print(f"    total {total:.1f} ms = {100 * total / CHUNK_MS:.1f}% of a chunk")

print("\nALL OK" if not fails else f"\n{fails} FAILED")
sys.exit(fails)
