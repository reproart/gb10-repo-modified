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
   torch._scaled_mm), and what each would cost a 4096-token chunk over the
   model's 36 GDN + 12 attention + 48 shared-expert layers.
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

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gb10_fp8_hc as hcp  # noqa: E402
import gb10_fp8_hc_kernel as hck  # noqa: E402
from gb10_fp8_side import quantize_per_channel  # noqa: E402

SECTIONS = {int(s) for s in os.environ.get("BENCH_SECTIONS", "1,2,3,4").split(",") if s}
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
        same = torch.equal(out, hck.fused_hc_mix_fp8(x, wd, sd, wu, su, HC, HS, num_ctas=SMS))
        ok = err < 1e-2 and same
        fails += not ok
        t8 = graph_us(lambda i: hck.fused_hc_mix_fp8(x, *copies[i][2:], HC, HS, num_ctas=SMS),
                      len(copies))
        line = (f"  {'PASS' if ok else 'FAIL'} {rows:3d} rows: FP8 {t8:6.1f} us "
                f"({fp8_bytes / t8 / 1e3:4.0f} GB/s), rel {err:.1e}, same bits: {same}")
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

    # (name, N, K, layers in the model)
    shapes = [("GDN in_proj_qkvz", 16384, 2560, 36), ("GDN in_proj_ba (padded)", 128, 2560, 36),
              ("GDN out_proj", 2560, 6144, 36), ("attn qkv_proj", 13312, 2560, 12),
              ("attn o_proj", 2560, 6144, 12), ("shared gate_up", 1280, 2560, 48),
              ("shared down", 2560, 640, 48)]
    names = ["marlin", "bf16", "dq+bf16"] + [f"w8a8-{k}" for k in w8a8]
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
        print(f"  {label} [{n} x {k}] x{count}:")
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
          + ", ".join(f"{k} {v / 1e3:.0f} ms" for k, v in chunk.items()))

print("\nALL OK" if not fails else f"\n{fails} FAILED")
sys.exit(fails)
