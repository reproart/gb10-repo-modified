#!/usr/bin/env python3
"""CPU checks for gb10_fp8_rowwise.py and patch_hc_fp8.py (torch; no GPU, no vLLM).

1. quantize_per_row: each row's largest weight maps to 448, every weight within E4M3's
   rounding of its row scale;
2. the torch (prefill) path equals x @ W.T for the dequantized weight, BF16 tail rows
   included;
3. with Triton installed, the GEMM kernel itself in Triton's interpreter
   (TRITON_INTERPRET=1, set here): FP8 rows + a tail of plain rows, split K and not,
   1-64 rows; two runs give the same bits; config() for the hyper-connection shapes.
   In FP16: the interpreter's tl.dot is wrong for BF16 (numpy has no bfloat16), and
   the kernel is the same for both; bench_fp8_rowwise_gpu.py checks BF16 on the GPU;
4. the custom op: torch path on CPU, the fake (meta) shape for torch.compile;
5. install_hc on stand-in vLLM classes: the down(+block_inject) and up layers get the
   FP8 method after __init__, a shared method object is left alone, loading converts
   them (block_inject rows stay BF16) and apply matches the BF16 layer, nothing is
   wrapped twice, other quant methods are left alone;
6. the patcher on a stand-in hyperconnection.py: hook appended once, inert without
   VLLM_HC_FP8, refuses a file without the hyper-connection layers.

    python3 src/test_hc_fp8_cpu.py        (kernel checks need: pip install triton)
"""
import os
import sys
import tempfile
import types

os.environ.setdefault("TRITON_INTERPRET", "1")
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gb10_fp8_rowwise as fr  # noqa: E402
import patch_hc_fp8 as ph  # noqa: E402

fails = 0


def check(name, ok):
    global fails
    fails += not ok
    print(("PASS " if ok else "FAIL ") + name)


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


g = torch.Generator().manual_seed(0)

# 1. quantization
w = (torch.randn(336, 1024, generator=g) * 0.02).to(torch.bfloat16)
w[5] = 0
q, s = fr.quantize_per_row(w)
deq = fr.dequantize(q, s)
amax = w.float().abs().amax(1)
err = (deq - w.float()).abs()
bound = torch.maximum(w.float().abs() * 2 ** -4, s[:, None] * 2 ** -9) * 1.001
check("quantize: E4M3 rows, float32 scales = amax / 448",
      q.dtype == torch.float8_e4m3fn and s.dtype == torch.float32 and torch.allclose(s[amax > 0], amax[amax > 0] / 448))
check("  each row's largest weight exact, every weight within E4M3 rounding", bool((err <= bound).all())
      and torch.allclose(deq.abs().amax(1)[amax > 0], amax[amax > 0], rtol=1e-6))
check("  an all-zero row stays zero (no NaN)", bool((deq[5] == 0).all()) and bool(torch.isfinite(s).all()))
check(f"  relative error of the weight {rel(deq, w):.4f} (< 0.03)", rel(deq, w) < 0.03)

# 2. torch path
q8, s8 = fr.quantize_per_row(w[:320])
tail = w[320:].contiguous()
x = torch.randn(70, 1024, generator=g).to(torch.bfloat16)
ref = x.float() @ fr.dequantize(q8, s8, tail).T
y = fr.torch_mm(x, q8, s8, tail)
check(f"torch path: FP8 rows scaled + BF16 tail = x @ W.T (rel {rel(y, ref):.4f})",
      y.shape == (70, 336) and y.dtype == torch.bfloat16 and rel(y, ref) < 1e-2)
check(f"  the tail columns BF16-exact (rel {rel(y[:, 320:], F.linear(x, tail)):.5f} to the BF16 layer's own)",
      rel(y[:, 320:], F.linear(x, tail)) < 5e-3)
wu = (torch.randn(256, 48, generator=g) * 0.02).to(torch.bfloat16)  # rows > K: scale on the weight
qu, su = fr.quantize_per_row(wu)
xu = torch.randn(70, 48, generator=g).to(torch.bfloat16)
yu = fr.torch_mm(xu, qu, su, wu[:0])
check(f"  more rows than K (up projection in prefill): scale on a weight copy (rel {rel(yu, xu.float() @ fr.dequantize(qu, su).T):.4f})",
      yu.shape == (70, 256) and rel(yu, xu.float() @ fr.dequantize(qu, su).T) < 1e-2)

# 3. the kernel, interpreted
try:
    import triton  # noqa: F401
    have_triton = True
except ImportError:
    have_triton = False
if have_triton:
    check("config: down + block_inject [336 x 10240] on 48 SMs -> 16-column blocks, K in 8 splits of 1280",
          fr.config(336, 10240, 320, 48) == (16, 256, 8, 1280))
    check("config: up [10240 x 320] -> 64-column blocks, no split", fr.config(10240, 320, 10240, 48) == (64, 64, 1, 320))
    cases = [  # rows, N, n8, K, cfg
        (4, 336, 320, 1024, (16, 256, 4, 256)),   # split K, BF16 tail
        (4, 336, 320, 1024, (16, 256, 1, 1024)),  # no split
        (1, 256, 256, 320, (64, 64, 1, 320)),     # no tail, K not a multiple of the split
        (20, 96, 96, 640, (16, 64, 3, 256)),      # 32-row tile, uneven last split
        (64, 48, 32, 512, (16, 128, 2, 256)),     # 64-row tile
    ]
    for m, n, n8, k, cfg in cases:
        wk = (torch.randn(n, k, generator=g) * 0.02).to(torch.bfloat16)
        qk, sk = fr.quantize_per_row(wk[:n8])
        tk = wk[n8:].contiguous().half()
        xk = torch.randn(m, k, generator=g).half()
        rk = xk.float() @ fr.dequantize(qk, sk, tk).T
        y1 = fr.triton_mm(xk, qk, sk, tk, cfg)
        y2 = fr.triton_mm(xk, qk, sk, tk, cfg)
        check(f"kernel {m} rows x [{n} ({n8} FP8) x {k}], cfg {cfg}: = reference (rel {rel(y1, rk):.4f}), "
              "same bits twice", y1.shape == (m, n) and rel(y1, rk) < 1e-2 and torch.equal(y1, y2))
    # FP16: bit for bit. BF16: within one BF16 step, because the interpreter truncates
    # FP32 -> BF16 where the GPU rounds to nearest like torch (the GPU bench checks the bits)
    for dt in (torch.float16, torch.bfloat16):
        dq = fr.dequant_weight(q8, s8, tail, dt, force_triton=True)
        ref_dq = fr.dequant_weight(q8.clone(), s8, tail, dt)
        close = torch.equal(dq, ref_dq) if dt == torch.float16 else bool(
            ((dq.float() - ref_dq.float()).abs() <= ref_dq.float().abs() * 2 ** -7).all())
        check(f"dequantize kernel ({str(dt)[6:]}): = the torch conversion, tail rows bit for bit",
              "dequant" not in fr._state["bad_tiles"] and dq.dtype == dt and close and torch.equal(dq[320:], tail.to(dt)))
    # the dequantize kernel built first (prefill runs first at boot) must not hide the GEMM's
    fr._kernels.clear()
    fr.dequant_weight(q8, s8, tail, torch.float16, force_triton=True)
    try:
        yk = fr.triton_mm(torch.randn(4, 1024, generator=g).half(), q8, s8, tail.half(), (16, 256, 4, 256))
        order_ok = yk.shape == (4, 336)
    except KeyError as e:
        order_ok = False
        print("   ", repr(e))
    check("  the GEMM kernel builds after the dequantize kernel (one kernel dict, keyed by name)", order_ok)
    xs = torch.randn(8, 1024, generator=g).half()
    one = fr.triton_mm(xs[:1], q8, s8, tail.half(), (16, 256, 4, 256))
    many = fr.triton_mm(xs, q8, s8, tail.half(), (16, 256, 4, 256))
    check("  a row's result does not depend on the other rows in the batch", torch.equal(one[0], many[0]))
else:
    print("SKIP the kernel checks: triton is not installed (pip install triton)")

# 4. the custom op
y = torch.ops.gb10.fp8_row_linear(x[:3].reshape(1, 3, 1024), q8, s8, tail)
check("custom op on CPU: the torch path, leading dims kept",
      y.shape == (1, 3, 336) and torch.equal(y.reshape(3, 336), fr.torch_mm(x[:3], q8, s8, tail)))
from torch._subclasses.fake_tensor import FakeTensorMode  # noqa: E402

x4 = x[:4].clone()
with FakeTensorMode() as mode:
    fy = torch.ops.gb10.fp8_row_linear(mode.from_tensor(x4), mode.from_tensor(q8),
                                       mode.from_tensor(s8), mode.from_tensor(tail))
check("  fake impl (torch.compile tracing): shape [4, 336], BF16", tuple(fy.shape) == (4, 336) and fy.dtype == torch.bfloat16)


# 5. install_hc on stand-ins
class LinearBase(torch.nn.Module):
    def __init__(self, n, k, method, output_sizes=None):
        super().__init__()
        self.weight = torch.nn.Parameter((torch.randn(n, k, generator=g) * 0.02).to(torch.bfloat16), requires_grad=False)
        self.quant_method = method
        if output_sizes:
            self.output_sizes = output_sizes

    def forward(self, x):
        return self.quant_method.apply(self, x, None)


class UnquantizedLinearMethod:
    def process_weights_after_loading(self, layer):
        layer._base_processed = True

    def apply(self, layer, x, bias=None):
        return F.linear(x, layer.weight, bias)


class OtherMethod(UnquantizedLinearMethod):
    pass


shared = UnquantizedLinearMethod()
ns = {"__name__": "hc_standin"}


class HyperConnection(torch.nn.Module):
    def __init__(self, inject=True):
        super().__init__()
        if inject:
            self.input_mix_weight_down_block_inject = LinearBase(336, 1024, shared, [320, 4, 12])
        else:
            self.input_mix_weight_down = LinearBase(320, 1024, UnquantizedLinearMethod())
        self.input_mix_weight_up = LinearBase(1024, 320, UnquantizedLinearMethod())
        self.router = LinearBase(64, 1024, UnquantizedLinearMethod())


class Mixer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_mix_weight_up = LinearBase(1024, 320, OtherMethod())


HyperConnection.__module__ = Mixer.__module__ = "hc_standin"
ns.update(HyperConnection=HyperConnection, Mixer=Mixer, LinearBase=LinearBase)
n_wrapped = fr.install_hc(ns, LinearBase, UnquantizedLinearMethod)
check("install_hc: both classes of the module wrapped", n_wrapped == 2)
try:
    fr.install_hc(ns, LinearBase, UnquantizedLinearMethod)
    twice = False
except RuntimeError:
    twice = True
check("  installing twice wraps nothing twice (says so)", twice)
hc = HyperConnection()
down, up = hc.input_mix_weight_down_block_inject, hc.input_mix_weight_up
check("  down(+inject) and up get the FP8 method; the router and the shared method object do not",
      type(down.quant_method).__name__ == "Gb10Fp8RowLinearMethod"
      and type(up.quant_method).__name__ == "Gb10Fp8RowLinearMethod"
      and type(hc.router.quant_method) is UnquantizedLinearMethod and type(shared) is UnquantizedLinearMethod)
check("  another quant method is left alone", type(Mixer().input_mix_weight_up.quant_method) is OtherMethod)
xd = torch.randn(4, 1024, generator=g).to(torch.bfloat16)
ref_down, ref_up = F.linear(xd, down.weight), F.linear(xd[:, :320], up.weight)
w_inject = down.weight.data[320:].clone()
fr.STATS.update(layers=0, bytes_before=0, bytes_after=0)
for lin in (down, up):
    lin.quant_method.process_weights_after_loading(lin)
check("  loading: the base method ran, weights FP8, block_inject rows (320-335) kept BF16 as they were",
      down._base_processed and down.weight.dtype == torch.float8_e4m3fn and down.weight.shape == (320, 1024)
      and torch.equal(down.gb10_tail, w_inject) and up.weight.shape == (1024, 320) and up.gb10_tail.shape[0] == 0)
check(f"  apply = the BF16 layer within FP8 rounding (down {rel(down(xd), ref_down):.4f}, up {rel(up(xd[:, :320]), ref_up):.4f})",
      rel(down(xd), ref_down) < 0.03 and rel(up(xd[:, :320]), ref_up) < 0.03
      and rel(down(xd)[:, 320:], F.linear(xd, w_inject)) < 5e-3)
check("  stats: 2 layers, ~half the bytes",
      fr.STATS["layers"] == 2 and 0.45 < fr.STATS["bytes_after"] / fr.STATS["bytes_before"] < 0.56)
hc2 = HyperConnection(inject=False)
check("  the plain down projection (no block_inject) converts whole",
      type(hc2.input_mix_weight_down.quant_method).__name__ == "Gb10Fp8RowLinearMethod")

# 6. the patcher
standin = '''import torch
from vllm.model_executor.layers.linear import ReplicatedLinear, LinearBase, UnquantizedLinearMethod


class HyperConnection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_mix_weight_down = ReplicatedLinear(10240, 320)
        self.input_mix_weight_up = ReplicatedLinear(320, 10240)
'''
lin_mod = types.ModuleType("vllm.model_executor.layers.linear")


class ReplicatedLinear(LinearBase):
    def __init__(self, k, n):
        super().__init__(n, k, UnquantizedLinearMethod())


lin_mod.ReplicatedLinear, lin_mod.LinearBase, lin_mod.UnquantizedLinearMethod = (
    ReplicatedLinear, LinearBase, UnquantizedLinearMethod)
for name in ("vllm", "vllm.model_executor", "vllm.model_executor.layers"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["vllm.model_executor.layers.linear"] = lin_mod
with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "hyperconnection.py")
    open(p, "w").write(standin)
    first, second = ph.patch(p), ph.patch(p)
    patched = open(p).read()
    check("patcher: hook appended once", first == "INSTALLED" and second == "already installed"
          and patched.count(ph.MARK) == 1)
    results = {}
    for env in ("0", "1"):
        os.environ["VLLM_HC_FP8"] = env
        mod = types.ModuleType(f"hc_patched_{env}")
        exec(compile(patched, p, "exec"), mod.__dict__)
        results[env] = type(mod.HyperConnection().input_mix_weight_up.quant_method).__name__
    os.environ.pop("VLLM_HC_FP8")
    check("  inert without VLLM_HC_FP8, FP8 method with VLLM_HC_FP8=1",
          results == {"0": "UnquantizedLinearMethod", "1": "Gb10Fp8RowLinearMethod"})
    open(p, "w").write("import torch\n")
    try:
        ph.patch(p)
        refused = False
    except SystemExit:
        refused = True
    check("  refuses a file without the hyper-connection layers", refused)

print("ALL OK" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
