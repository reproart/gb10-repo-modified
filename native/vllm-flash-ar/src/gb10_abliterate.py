"""Refusal-direction ablation at inference for Qwen3.8-Flash-Next on vLLM (qwen38-flash-dgx
image; vLLM 0.34 in ../vllm-flash-v034). src/patch_abliterate.py wires it into
models/qwen3_8_flash_next/nvidia/model.py and mtp.py (0.34: models/qwen4_exp/nvidia/) at build
time; this module is imported only when VLLM_ABLATE is set at runtime.

Where it acts. This model keeps hc_count = 4 residual streams per token ([T, 4 x 2560]), and
no layer writes them directly: a block (attention or GDN, then the MLP / MoE) returns one
[T, 2560] output, and GatedResidual's combine adds it to every stream with one scalar weight
per stream and token (2 sigmoid(injection logit / 4): hc_combine in ops/hc.py; the MLP's output
one layer later, the delayed combine). The other writers are the token embedding (repeated
into the 4 streams), the PLE layer's addition ([T, 4 x 2560], one layer) and the vision
tower's deepstack embeddings (images only). The elementwise gate along the hidden dimension
sits on the read side only (mix: per-element norm weight and sigmoid gate, then the mean over
the streams; the final mixer the same).

So taking a direction r out of every writer, x - (x . r) r, keeps r out of all four streams
for good: the same as orthogonalizing the writers' weights against r (Arditi et al. 2024,
"abliteration"), done on the fly, two small GEMVs a layer. alpha scales it,
x - alpha (x . r) r, and is folded into the stored vector (u = sqrt(alpha) r), so no Python
float is baked into the compiled graph. A rank-k basis B (orthonormal rows) removes a
subspace the same way: x - alpha B^T (B x).

Environment (the launcher: ABLATE, ABLATE_VECTORS, ... in config.sh):
  VLLM_ABLATE=<file>        the vectors, in the container. Off when unset.
  VLLM_ABLATE_SOURCE        which direction of a per-layer file:
                              mean    the normalized mean of its unit vectors (default)
                              <n>     layer n's vector
                              top<k>  its k principal directions (rank-k projection)
                            A single vector (.npy / .pt of shape [hidden]) is used as is.
  VLLM_ABLATE_ALPHA         1 (default) removes the component; 0 = the stock model; above 1
                            overshoots (msuiche's 27B card: 2 refuses harmless prompts, 4
                            breaks the model).
  VLLM_ABLATE_LAYERS        all (default) or a list like 4-46,47: whose writers are
                            projected. The embedding goes with layer 0, the PLE addition
                            and deepstack embeddings with the layer they are added at.
  VLLM_ABLATE_MTP           1 (default): the MTP draft layer projects its writers too (the
                            fused input fc_embedding + fc_hidden and its blocks), as a
                            checkpoint orthogonalized as a whole would; 0 = the draft as is.
                            Answers never depend on it (the target verifies every drafted
                            token), only the draft's acceptance.

Files: .npz with one vector per layer keyed by the layer index ("4", "5", ...) or a
[n, hidden] matrix next to a "layers" array (Mambavtt/qwen3.8-flash-next-refusal-ablation-
vectors, refusal_directions-v4-late.npz: layers 4-46); .npy / .pt holding [hidden] or
[n, hidden] (rows = layers 0..n-1) or, for .pt, a {layer: vector} dict.

Those per-layer vectors were taken from the mean of the four streams after each layer
(stream space, the space projected here). Their adjacent cosines are 0.91-0.99 and the
normalized mean has cosine 0.61-0.94 with each (layer 23 the most central); the log prints
how much of each layer's direction the chosen basis covers.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import math
import os

try:
    from vllm.logger import init_logger

    logger = init_logger("vllm.gb10_abliterate")
except ImportError:  # CPU tests
    logger = logging.getLogger("vllm.gb10_abliterate")

EMBEDDING = -1   # attach()'s layer index for the model's embedding


# --------------------------------------------------------------------------- settings
def parse_layers(spec: str):
    """'all' / '' -> None (every layer); '4-46,47' -> {4, ..., 47}."""
    spec = (spec or "all").strip().lower()
    if spec == "all":
        return None
    out = set()
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        lo, sep, hi = part.partition("-")
        try:
            a = int(lo)
            b = int(hi) if sep else a
        except ValueError:
            raise RuntimeError(f"VLLM_ABLATE_LAYERS={spec!r}: 'all' or a list like 4-46,47") from None
        if b < a:
            raise RuntimeError(f"VLLM_ABLATE_LAYERS={spec!r}: {part} runs backwards")
        out.update(range(a, b + 1))
    return out


def settings(env=None) -> dict | None:
    env = os.environ if env is None else env
    path = env.get("VLLM_ABLATE", "").strip()
    if not path:
        return None
    try:
        alpha = float(env.get("VLLM_ABLATE_ALPHA", "1") or 1)
    except ValueError:
        raise RuntimeError(f"VLLM_ABLATE_ALPHA={env.get('VLLM_ABLATE_ALPHA')!r}: a number") from None
    mtp = env.get("VLLM_ABLATE_MTP", "1").strip()
    if mtp not in ("0", "1"):
        raise RuntimeError(f"VLLM_ABLATE_MTP={mtp!r}: 1 or 0")
    return {
        "path": path,
        "source": (env.get("VLLM_ABLATE_SOURCE", "mean") or "mean").strip().lower(),
        "alpha": alpha,
        "layers": parse_layers(env.get("VLLM_ABLATE_LAYERS", "all")),
        "mtp": mtp == "1",
    }


# --------------------------------------------------------------------------- vectors
def read_vectors(path: str):
    """-> ({layer: float64 vector}, None) for a per-layer file, (None, vector) for one vector."""
    import numpy as np

    if not os.path.isfile(path):
        raise RuntimeError(f"VLLM_ABLATE={path}: no such file (in the container: the launcher "
                           "mounts ABLATE_VECTORS' directory at /ablate)")

    def from_matrix(mat, labels=None):
        mat = np.asarray(mat, dtype=np.float64)
        if mat.ndim == 1:
            return None, mat
        if mat.ndim != 2:
            raise RuntimeError(f"VLLM_ABLATE: {path}: an array of shape {mat.shape}, not [hidden] "
                               "or [layers, hidden]")
        idx = list(range(mat.shape[0])) if labels is None else [int(i) for i in np.asarray(labels).reshape(-1)]
        if len(idx) != mat.shape[0]:
            raise RuntimeError(f"VLLM_ABLATE: {path}: {len(idx)} layer indices for {mat.shape[0]} rows")
        return dict(zip(idx, mat)), None

    if path.endswith(".npz"):
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        keyed = {int(k): np.asarray(a, dtype=np.float64) for k, a in arrays.items() if k.isdigit()}
        if keyed:
            return keyed, None
        mats = {k: a for k, a in arrays.items() if k != "layers" and np.asarray(a).ndim in (1, 2)}
        if len(mats) == 1:
            return from_matrix(next(iter(mats.values())), arrays.get("layers"))
        raise RuntimeError(f"VLLM_ABLATE: no vectors in {path}: entries "
                           + ", ".join(f"{k} {tuple(a.shape)}" for k, a in arrays.items()))
    if path.endswith(".npy"):
        return from_matrix(np.load(path, allow_pickle=False))
    if path.endswith(".pt") or path.endswith(".pth"):
        import torch

        obj = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(obj, dict):
            if "directions" in obj:
                return from_matrix(obj["directions"].float().numpy(), obj.get("layers"))
            return {int(k): v.float().numpy().astype(np.float64) for k, v in obj.items()}, None
        return from_matrix(obj.float().numpy())
    raise RuntimeError(f"VLLM_ABLATE={path}: .npz, .npy or .pt")


def basis(path: str, source: str, hidden: int):
    """-> (float64 [k, hidden] orthonormal rows, description, coverage {layer: |B d|^2})."""
    import numpy as np

    per_layer, single = read_vectors(path)

    def unit(v, what):
        v = np.asarray(v, dtype=np.float64).reshape(-1)
        if v.shape[0] != hidden:
            raise RuntimeError(f"VLLM_ABLATE: {what} in {path} has {v.shape[0]} elements; the model's "
                               f"hidden size is {hidden}")
        n = np.linalg.norm(v)
        if not np.isfinite(n) or n == 0:
            raise RuntimeError(f"VLLM_ABLATE: {what} in {path} is zero or not finite")
        return v / n

    if single is not None:
        return unit(single, "the vector")[None, :], "the file's one vector", {}
    layers = sorted(per_layer)
    U = np.stack([unit(per_layer[i], f"layer {i}'s vector") for i in layers])
    if source == "mean":
        m = U.mean(axis=0)
        if np.linalg.norm(m) < 0.1:
            raise RuntimeError(f"VLLM_ABLATE: the layers' vectors in {path} nearly cancel (|mean| "
                               f"{np.linalg.norm(m):.3f}); pick one with VLLM_ABLATE_SOURCE=<layer>")
        B = (m / np.linalg.norm(m))[None, :]
        what = f"the mean of layers {layers[0]}-{layers[-1]}'s vectors"
    elif source.isdigit():
        if int(source) not in per_layer:
            raise RuntimeError(f"VLLM_ABLATE_SOURCE={source}: no such layer in {path} "
                               f"({layers[0]}-{layers[-1]})")
        B = U[layers.index(int(source))][None, :]
        what = f"layer {source}'s vector"
    elif source.startswith("top") and source[3:].isdigit() and 1 <= int(source[3:]) <= len(layers):
        k = int(source[3:])
        _, s, vt = np.linalg.svd(U, full_matrices=False)
        B = vt[:k]                                  # a projection does not care about signs
        share = (s[:k] ** 2).sum() / (s ** 2).sum()
        what = f"the top {k} principal directions of layers {layers[0]}-{layers[-1]}'s vectors ({share:.0%} of their energy)"
    else:
        raise RuntimeError(f"VLLM_ABLATE_SOURCE={source!r}: mean, a layer index or top<k>")
    B, _ = np.linalg.qr(B.T)                       # orthonormal rows, whatever came in
    B = B.T
    coverage = {i: float(np.sum((B @ U[j]) ** 2)) for j, i in enumerate(layers)}
    return B, what, coverage


# --------------------------------------------------------------------------- the projection
FUSED = os.environ.get("VLLM_ABLATE_FUSED", "1") != "0"   # one Triton kernel per projection on the GPU
_fused_state = {"bad": False, "logged": False}


def project_torch(x, u):
    """x [..., s * H] (s streams, s >= 1), u [k, H] = sqrt(alpha) * orthonormal rows:
    each H-wide stream loses alpha times its component in u's span. Plain torch: a
    matrix-vector product, an outer product and a subtraction, each reading or writing x's
    size again."""
    import torch

    u = u.to(x.dtype)
    xs = x.reshape(*x.shape[:-1], -1, u.shape[-1])             # [..., s, H]
    return (xs - torch.matmul(torch.matmul(xs, u.t()), u)).reshape(x.shape)


def project(x, u):
    """project_torch's result; on the GPU (VLLM_ABLATE_FUSED=1, the default) one fused Triton
    kernel (gb10::abl_project): each H-wide row read once, its k dot products and the
    subtraction in fp32, written once. Deterministic (no atomics, fixed reduction order)."""
    if FUSED and x.is_cuda and not _fused_state["bad"]:
        _register_op()
        import torch

        return torch.ops.gb10.abl_project(x, u)
    return project_torch(x, u)


_kernels = {}


def _get_kernel():
    if "k" in _kernels:
        return _kernels["k"]
    import triton
    import triton.language as tl

    @triton.jit
    def _abl_project_kernel(x_ptr, u_ptr, y_ptr, R, H, stride_x, stride_y,
                            K: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_H: tl.constexpr):
        rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
        cols = tl.arange(0, BLOCK_H)
        cmask = cols < H
        mask = (rows[:, None] < R) & cmask[None, :]
        x = tl.load(x_ptr + rows[:, None] * stride_x + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        acc = x
        for j in tl.static_range(K):
            u = tl.load(u_ptr + j * H + cols, mask=cmask, other=0.0).to(tl.float32)
            d = tl.sum(x * u[None, :], axis=1)            # the row's component along u_j
            acc = acc - d[:, None] * u[None, :]
        tl.store(y_ptr + rows[:, None] * stride_y + cols[None, :], acc.to(y_ptr.dtype.element_ty), mask=mask)

    _kernels["k"] = (_abl_project_kernel, triton)
    return _kernels["k"]


def fused_project(x, u):
    """The fused kernel on x's device (CUDA, or Triton's interpreter in the CPU tests)."""
    import torch

    kern, triton = _get_kernel()
    k, h = u.shape
    x2 = x.reshape(-1, h)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    y = torch.empty((x2.shape[0], h), dtype=x.dtype, device=x.device)
    if x2.shape[0]:
        block_h = triton.next_power_of_2(h)
        block_r = 1 if block_h >= 4096 else 2
        kern[(triton.cdiv(x2.shape[0], block_r),)](
            x2, u.contiguous(), y, x2.shape[0], h, x2.stride(0), y.stride(0),
            K=k, BLOCK_R=block_r, BLOCK_H=block_h, num_warps=8 if block_h >= 2048 else 4)
    return y.reshape(x.shape)


def _register_op():
    import torch

    if hasattr(torch.ops, "gb10") and hasattr(torch.ops.gb10, "abl_project"):
        return

    # an explicit schema: this module's annotations are strings (from __future__ import annotations)
    @torch.library.custom_op("gb10::abl_project", mutates_args=(), schema="(Tensor x, Tensor u) -> Tensor")
    def abl_project(x, u):
        if x.is_cuda and not _fused_state["bad"]:
            try:
                y = fused_project(x, u)
                if not _fused_state["logged"]:
                    _fused_state["logged"] = True
                    logger.warning("ABLATION: projections on the fused kernel (gb10::abl_project; "
                                   "VLLM_ABLATE_FUSED=0 for the torch ops)")
                return y
            except Exception as e:  # noqa: BLE001 - the torch ops, slower, give the same projection
                _fused_state["bad"] = True
                logger.warning("ABLATION: the fused projection kernel failed (%s: %s); torch ops from now on",
                               type(e).__name__, e)
        return project_torch(x, u)

    @abl_project.register_fake
    def _(x, u):
        return torch.empty_like(x)


# --------------------------------------------------------------------------- wiring
class Ablation:
    """One install: the settings, the basis and the shared vectors (one tensor per dtype /
    device, registered as a non-persistent buffer in every module that projects)."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.basis = None
        self.vectors = {}

    def vector(self, hidden: int):
        import numpy as np
        import torch

        if self.basis is None:
            B, what, coverage = basis(self.cfg["path"], self.cfg["source"], hidden)
            self.basis, self.what, self.coverage = B, what, coverage
            with open(self.cfg["path"], "rb") as f:
                self.sha = hashlib.sha256(f.read()).hexdigest()
        if self.basis.shape[1] != hidden:
            raise RuntimeError(f"VLLM_ABLATE: basis width {self.basis.shape[1]} != hidden {hidden}")
        dtype, device = torch.get_default_dtype(), torch.empty(0).device
        key = (dtype, str(device))
        if key not in self.vectors:
            u = torch.from_numpy(np.ascontiguousarray(self.basis * math.sqrt(self.cfg["alpha"])))
            self.vectors[key] = u.to(device=device, dtype=dtype)
        return self.vectors[key]

    def prewarm(self, u) -> None:
        """Build the fused kernel now, at model build, not inside a CUDA graph capture; a
        failure here switches to the torch ops, loudly. Once."""
        if getattr(self, "_prewarmed", False) or not (FUSED and u.is_cuda):
            return
        self._prewarmed = True
        try:
            import torch

            x = torch.zeros((2, u.shape[1]), dtype=u.dtype, device=u.device)
            fused_project(x, u)
            torch.cuda.synchronize(u.device)
        except Exception as e:  # noqa: BLE001
            _fused_state["bad"] = True
            logger.warning("ABLATION: the fused projection kernel does not build here (%s: %s); torch ops",
                           type(e).__name__, e)

    def wants(self, layer: int, n_layers: int) -> bool:
        if layer >= n_layers:
            return self.cfg["mtp"]
        sel = self.cfg["layers"]
        if layer == EMBEDDING:
            return sel is None or 0 in sel
        return sel is None or layer in sel

    def attach(self, module, layer: int, config) -> bool:
        n_layers, hidden = int(config.num_hidden_layers), int(config.hidden_size)
        on = self.cfg["alpha"] != 0 and self.wants(layer, n_layers)
        module._gb10_abl_on = on
        if on:
            u = self.vector(hidden)
            module.register_buffer("_gb10_abl_u", u, persistent=False)
            self.prewarm(u)
        if layer >= n_layers:
            logger.warning("ABLATION: MTP draft layer %d %s", layer,
                           "projects its writers too" if on else "left as is (VLLM_ABLATE_MTP=0)")
        return on

    def report(self, model) -> None:
        """One line per model built: what this instance projects."""
        cfg = self.cfg
        n_layers = int(model.config.num_hidden_layers)
        if cfg["alpha"] == 0:
            logger.warning("ABLATION: VLLM_ABLATE_ALPHA=0, the stock model (nothing projected)")
            return
        built = [m for m in model.layers if hasattr(m, "layer_idx")]   # not pipeline stand-ins
        layers = sorted(m.layer_idx for m in built if getattr(m, "_gb10_abl_on", False))
        skipped = sorted(m.layer_idx for m in built if not getattr(m, "_gb10_abl_on", False))
        span = f"{layers[0]}-{layers[-1]}" if layers and layers == list(range(layers[0], layers[-1] + 1)) \
            else ",".join(map(str, layers)) or "none"
        cov = ""
        if self.coverage:
            vals = list(self.coverage.values())
            cov = f"; covers {min(vals):.0%}-{max(vals):.0%} (mean {sum(vals) / len(vals):.0%}) of the file's per-layer directions"
        logger.warning(
            "ABLATION ON (VLLM_ABLATE): refusal direction %s from %s (sha256 %s...), rank %d, alpha %g%s; "
            "projected out of the writers of %d of %d layers (%s)%s%s. Answers are not the stock model's.",
            self.what, os.path.basename(cfg["path"]), self.sha[:12], self.basis.shape[0], cfg["alpha"], cov,
            len(layers), n_layers, span, ", the embedding" if getattr(model, "_gb10_abl_on", False) else "",
            "" if not skipped else f"; layers {','.join(map(str, skipped))} untouched")
        if cfg["alpha"] > 1:
            logger.warning("ABLATION: alpha %g > 1 overshoots: the component is reflected, not removed "
                           "(msuiche's 27B card: 2 refuses harmless prompts, 4 breaks the model)", cfg["alpha"])
        if cfg["layers"] is not None:
            extra = sorted(i for i in cfg["layers"] if not 0 <= i < n_layers)
            if extra:
                logger.warning("ABLATION: VLLM_ABLATE_LAYERS names layers %s, not in 0-%d", extra, n_layers - 1)


def install(ns: dict, layer: str = "Qwen3_8FlashNextDecoderLayer", model: str = "Qwen3_8FlashNextModel"):
    """Called from model.py's appended hook with its globals and its two class names (vLLM 0.34:
    Qwen4ExpDecoderLayer, Qwen4ExpModel): wraps the decoder layer's and the model's __init__ so
    every instance attaches (or not) after it is built. The forward code the patch inserted
    reads the flags and buffers."""
    layer_cls = ns[layer]
    model_cls = ns[model]
    if getattr(layer_cls, "_gb10_abl", None) is not None:
        return layer_cls._gb10_abl
    cfg = settings()
    if cfg is None:
        return None
    abl = Ablation(cfg)
    if not os.path.isfile(cfg["path"]):
        read_vectors(cfg["path"])                   # raises with the message

    layer_init = layer_cls.__init__

    @functools.wraps(layer_init)
    def init_layer(self, *args, **kwargs):
        layer_init(self, *args, **kwargs)
        abl.attach(self, self.layer_idx, self.config)

    model_init = model_cls.__init__

    @functools.wraps(model_init)
    def init_model(self, *args, **kwargs):
        model_init(self, *args, **kwargs)
        abl.attach(self, EMBEDDING, self.config)
        abl.report(self)

    layer_cls.__init__ = init_layer
    model_cls.__init__ = init_model
    layer_cls._gb10_abl = abl
    logger.warning("ABLATION: VLLM_ABLATE=%s (source %s, alpha %g, layers %s, MTP %s): armed",
                   cfg["path"], cfg["source"], cfg["alpha"],
                   "all" if cfg["layers"] is None else os.environ.get("VLLM_ABLATE_LAYERS"),
                   "yes" if cfg["mtp"] else "no")
    return abl
