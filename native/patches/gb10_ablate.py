"""Directional ablation (refusal-direction vectors) for Qwen3.8-Flash-Next at load.

For per-layer unit directions d_l (an .npz with one float vector per layer,
keys = layer indices as strings, e.g. the "refusal_directions-v4-late.npz"
vectors built for RadixArk/Qwen3.8-Flash-Next-NVFP4), every listed layer's
residual stream is projected at inference time:

    h' = h - alpha * d_l * (d_l . h)

No weight changes: alpha 0 is the stock model, 1 the vectors' validated
default, more is stronger (watch for incoherence). Enabled by
GB10_ABLATE=<path to .npz> (models/qwen3.8-flash-next.sh: ABLATE=...).

Where. Qwen4-Exp passes one tensor between decoder layers: [tokens,
hc_count * hidden] (4 hyper-connection streams of 2560), the residual already
combined in (each layer returns (hidden_states, None)). The vectors were
taken from the mean over the 4 streams, so the projection is applied
  GB10_ABLATE_STREAMS=each (default): to each stream; the direction leaves
                     every stream and so the mean too;
  GB10_ABLATE_STREAMS=mean: only the streams' common component along d
                     (the mean's), leaving their differences along d.
GB10_ABLATE_AT=output (default) projects layer l's output, =input its input
(= layer l-1's output). Which one the vectors were captured at is not
stated; adjacent layers' directions are logged with their cosine, and when
that is high the choice barely matters.

The MTP draft is not touched; it reads the ablated target states, so its
accept_len may move a little. The prefix cache lives in memory, so a restart
with another alpha never mixes cached states. Cost: two small kernels per
ablated layer per forward (a GEMV and a rank-1 update), ~0.5 ms a decode step
for 43 layers. Written against SGLang 0.5.20.
"""

from __future__ import annotations

import hashlib
import logging
import os

logger = logging.getLogger("sglang.srt.models.qwen4_exp.gb10_ablate")

TARGET_MODULE = "sglang.srt.models.qwen4_exp"
TARGET_CLASS = "Qwen4ExpForConditionalGeneration"


def settings() -> dict:
    path = os.environ.get("GB10_ABLATE", "")
    alpha = float(os.environ.get("GB10_ABLATE_ALPHA", "1.0"))
    at = os.environ.get("GB10_ABLATE_AT", "output")
    streams = os.environ.get("GB10_ABLATE_STREAMS", "each")
    if at not in ("output", "input"):
        raise RuntimeError(f"GB10_ABLATE_AT must be output or input, not {at!r}")
    if streams not in ("each", "mean"):
        raise RuntimeError(f"GB10_ABLATE_STREAMS must be each or mean, not {streams!r}")
    return {"path": path, "alpha": alpha, "at": at, "streams": streams}


def _unit(v, layer, hidden):
    import numpy as np

    v = np.asarray(v, dtype=np.float32).reshape(-1)
    if v.shape[0] != hidden:
        raise RuntimeError(f"GB10_ABLATE: layer {layer} vector has {v.shape[0]} values, "
                           f"the model's hidden size is {hidden}: vectors for another model?")
    n = float(np.linalg.norm(v))
    if not n > 0:
        raise RuntimeError(f"GB10_ABLATE: layer {layer} vector is zero")
    if abs(n - 1.0) > 1e-3:
        logger.info("GB10_ABLATE: layer %s vector norm %.4f, normalized", layer, n)
    return v / n


def load_directions(path: str, hidden: int) -> dict:
    """{layer index: float32 numpy unit vector [hidden]} from an .npz in
    either layout:
      * one array per layer, keyed by the layer index ("4", "5", ...); other
        keys (metadata such as a "layers" list) are listed and skipped;
      * one [n, hidden] matrix next to a 1-D "layers" array of n indices."""
    import numpy as np

    if not os.path.isfile(path):
        raise RuntimeError(f"GB10_ABLATE={path}: no such file")
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    out, other = {}, {}
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    for key, arr in arrays.items():
        if key.isdigit():
            out[int(key)] = _unit(arr, key, hidden)
        else:
            other[key] = arr
    listed = other.get("layers")
    if not out:
        mats = {k: a for k, a in other.items() if a.ndim == 2 and a.shape[1] == hidden}
        if listed is not None and len(mats) == 1:
            name, mat = next(iter(mats.items()))
            idx = [int(i) for i in np.asarray(listed).reshape(-1)]
            if len(idx) != mat.shape[0]:
                raise RuntimeError(f"GB10_ABLATE: {len(idx)} layer indices for {mat.shape[0]} "
                                   f"rows of {name!r} in {path}")
            out = {i: _unit(row, i, hidden) for i, row in zip(idx, mat)}
            other = {k: a for k, a in other.items() if k not in (name, "layers")}
    elif listed is not None:
        idx = sorted(int(i) for i in np.asarray(listed).reshape(-1))
        if idx != sorted(out):
            logger.warning("GB10_ABLATE: the file's 'layers' list %s differs from its vector "
                           "keys %s; using the keys", idx, sorted(out))
        other.pop("layers")
    if other:
        logger.info("GB10_ABLATE: skipped non-vector entries %s",
                    ", ".join(f"{k} {tuple(a.shape)}" for k, a in other.items()))
    if not out:
        raise RuntimeError(f"GB10_ABLATE: no per-layer vectors in {path}: entries " + ", ".join(
            f"{k} {tuple(a.shape)}" for k, a in arrays.items()))
    logger.info("GB10_ABLATE: %s (sha256 %s...), %d layers %d..%d", os.path.basename(path),
                digest[:12], len(out), min(out), max(out))
    return out


def adjacent_cosines(dirs: dict) -> list:
    import numpy as np

    keys = sorted(dirs)
    return [float(np.dot(dirs[a], dirs[b])) for a, b in zip(keys, keys[1:]) if b == a + 1]


def project(h, d, alpha: float, hidden: int, streams: str):
    """h [..., k * hidden] (k streams) -> h with alpha * its d-component removed."""
    import torch

    shape = h.shape
    if shape[-1] % hidden:
        raise RuntimeError(f"GB10_ABLATE: hidden state width {shape[-1]} is not a multiple "
                           f"of {hidden}")
    x = h.reshape(-1, hidden)                      # [tokens * k, hidden]
    coef = torch.mv(x, d)                          # [tokens * k]
    if streams == "mean":
        k = shape[-1] // hidden
        coef = coef.view(-1, k).mean(dim=1, keepdim=True).expand(-1, k).reshape(-1)
    return torch.addr(x, coef, d, alpha=-alpha).view(shape)


def find_text_model(model):
    """The Qwen4ExpModel inside the target: the module with the decoder
    layers (layers, start_layer, end_layer, hc_count)."""
    for m in model.modules():
        if all(hasattr(m, a) for a in ("layers", "start_layer", "end_layer", "hc_count")):
            return m
    raise RuntimeError("GB10_ABLATE: no decoder layer stack (layers/start_layer/hc_count) "
                       "in the target model (written for SGLang 0.5.20)")


def install(model, cfg: dict) -> list:
    """Register the hooks on `model` (after its weights are loaded). Returns
    the hook handles."""
    import torch

    if cfg["alpha"] == 0:
        logger.info("GB10_ABLATE: alpha 0, the stock model; no hooks")
        return []
    text = find_text_model(model)
    hidden = int(getattr(text, "hidden_size", 0) or text.config.hidden_size)
    dirs = load_directions(cfg["path"], hidden)
    n_layers = len(text.layers)
    bad = [i for i in dirs if not 0 <= i < n_layers]
    if bad:
        raise RuntimeError(f"GB10_ABLATE: layers {bad} not in the model's 0..{n_layers - 1}")
    alpha, streams = cfg["alpha"], cfg["streams"]
    handles = []
    for i in sorted(dirs):
        if not text.start_layer <= i < text.end_layer:
            continue
        layer = text.layers[i]
        # a CUDA parameter's device: PLE tables can live in pinned host memory
        p = next((q for q in layer.parameters() if q.is_cuda), None)
        device = p.device if p is not None else (
            torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available()
            else torch.device("cpu"))
        # float32 on the device, cast once per state dtype (bf16 in serving)
        d32 = torch.from_numpy(dirs[i]).to(device=device)
        # bf16/fp16 cast here, not inside a CUDA graph capture
        cast = {torch.float32: d32, torch.bfloat16: d32.bfloat16(), torch.float16: d32.half()}

        def d_for(dtype, cast=cast, d32=d32):
            if dtype not in cast:
                cast[dtype] = d32.to(dtype)
            return cast[dtype]

        if cfg["at"] == "output":
            def hook(_m, _args, output, d_for=d_for, i=i):
                hs, residual = output
                if residual is not None:
                    raise RuntimeError(f"GB10_ABLATE: layer {i} returned a separate residual; "
                                       "this SGLang's Qwen4-Exp layout differs from 0.5.20")
                return project(hs, d_for(hs.dtype), alpha, hidden, streams), residual
            handles.append(layer.register_forward_hook(hook))
        else:
            def pre_hook(_m, args, kwargs, d_for=d_for, i=i):
                if kwargs.get("residual") is not None:
                    raise RuntimeError(f"GB10_ABLATE: layer {i} got a separate residual; "
                                       "this SGLang's Qwen4-Exp layout differs from 0.5.20")
                hs = kwargs["hidden_states"]
                kwargs["hidden_states"] = project(hs, d_for(hs.dtype), alpha, hidden, streams)
                return args, kwargs
            handles.append(layer.register_forward_pre_hook(pre_hook, with_kwargs=True))
    cos = adjacent_cosines(dirs)
    logger.info(
        "GB10_ABLATE: %d layers, alpha %g, at layer %s, %s stream%s; adjacent directions' "
        "cosine %s", len(handles), alpha, cfg["at"], streams, "s" if streams == "each" else "s' mean",
        f"min {min(cos):.3f} / mean {sum(cos) / len(cos):.3f}" if cos else "n/a")
    return handles


def _wrap_load_weights(cls) -> None:
    orig = cls.load_weights

    def load_weights(self, weights, *args, **kwargs):
        result = orig(self, weights, *args, **kwargs)
        self._gb10_ablate_hooks = install(self, settings())
        return result

    cls.load_weights = load_weights


def apply_target(mod) -> None:
    cls = getattr(mod, TARGET_CLASS, None)
    if cls is None:
        raise RuntimeError(f"GB10_ABLATE: {TARGET_MODULE}.{TARGET_CLASS} not found "
                           "(written for SGLang 0.5.20); unset GB10_ABLATE.")
    settings()  # fail on a bad setting before the weights load
    _wrap_load_weights(cls)
    logger.info("GB10_ABLATE: %s will be ablated after loading", TARGET_CLASS)
