"""Directional ablation (refusal-direction vectors) at load, for Qwen3.8-Flash-Next
(Qwen4-Exp) and Qwen3.8-27B (Qwen3.5).

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

Qwen3.8-27B (sglang.srt.models.qwen3_5, SGLang 0.5.21). A decoder layer returns
only its MLP's contribution; the residual stream lives in the batch's
ResidualStream (forward_batch.residual_stream: .residual, the stream up to this
layer's MLP, and .pending.value, the very tensor the layer returned), and the
next layer's boundary adds the two. The stream after layer l is residual + out,
so its projection is applied to the contribution in place:

    out' = out - alpha * d_l * (d_l . (residual + out))

and the next layer adds residual + out' = the projected stream (the same tensor
object, as SGLang checks). GB10_ABLATE_AT=input applies d_l at layer l-1's
output instead (= layer l's input). The DFlash draft reads the target states the
next layer captures, after the projection. The vectors may also be a GGUF
control vector in the "glp" projective format (glp.mode = project, tensors
direction.<layer>, e.g. msuiche/Qwen3.8-27B-abliterated-cyber-GLP-49): it says
h -= alpha (h . d) d on the post-layer residual stream, which is this. An
additive llama.cpp control vector (no glp.mode) is refused: adding a direction
is another operation. Their card: alpha 1 removes the component, 2 reflects it
(harmless prompts refused), 4 collapses the model.

The MTP / DFlash draft is not touched; it reads the ablated target states, so its
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
QWEN3_5_MODULE = "sglang.srt.models.qwen3_5"
QWEN3_5_CLASSES = ("Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM")


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


def read_gguf(path: str):
    """(metadata dict, {tensor name: float32 numpy array}) of a GGUF file (v2/v3;
    F32 tensors only, which is what control vectors hold)."""
    import struct

    import numpy as np

    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"GGUF":
        raise RuntimeError(f"GB10_ABLATE: {path} is not a GGUF file")
    pos = [4]

    def take(fmt):
        v = struct.unpack_from("<" + fmt, data, pos[0])
        pos[0] += struct.calcsize("<" + fmt)
        return v[0]

    def string():
        n = take("Q")
        s = data[pos[0]:pos[0] + n].decode("utf-8")
        pos[0] += n
        return s

    scalar = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

    def value(t):
        if t in scalar:
            return take(scalar[t])
        if t == 8:
            return string()
        if t == 9:
            item, n = take("I"), take("Q")
            return [value(item) for _ in range(n)]
        raise RuntimeError(f"GB10_ABLATE: {path}: unknown GGUF value type {t}")

    version = take("I")
    if version not in (2, 3):
        raise RuntimeError(f"GB10_ABLATE: {path}: GGUF version {version}, expected 2 or 3")
    n_tensors, n_kv = take("Q"), take("Q")
    meta = {}
    for _ in range(n_kv):
        key = string()
        meta[key] = value(take("I"))
    infos = []
    for _ in range(n_tensors):
        name = string()
        dims = [take("Q") for _ in range(take("I"))]
        infos.append((name, dims, take("I"), take("Q")))
    align = int(meta.get("general.alignment", 32))
    start = -(-pos[0] // align) * align
    tensors = {}
    for name, dims, typ, off in infos:
        if typ != 0:
            raise RuntimeError(f"GB10_ABLATE: {path}: tensor {name} is GGUF type {typ}, not F32")
        n = int(np.prod(dims)) if dims else 1
        tensors[name] = np.frombuffer(data, dtype="<f4", count=n, offset=start + off).reshape(dims[::-1])
    return meta, tensors


def load_gguf_directions(path: str, hidden: int) -> dict:
    """{layer: unit vector} from a projective ("glp") GGUF control vector."""
    meta, tensors = read_gguf(path)
    mode = meta.get("glp.mode")
    if mode != "project":
        raise RuntimeError(
            f"GB10_ABLATE: {path} has glp.mode={mode!r}, not 'project': an additive "
            "control vector (h + s d) is not a direction to ablate; refusing it")
    hook = meta.get("glp.hook_point", meta.get("glp.derived_at"))
    if hook not in (None, "residual_stream_post_layer"):
        logger.warning("GB10_ABLATE: %s is for hook point %r; applied at the post-layer "
                       "residual stream (GB10_ABLATE_AT) anyway", os.path.basename(path), hook)
    out = {}
    for name, arr in tensors.items():
        head, _, idx = name.rpartition(".")
        if head != "direction" or not idx.isdigit():
            raise RuntimeError(f"GB10_ABLATE: {path}: unexpected tensor {name!r}")
        out[int(idx)] = _unit(arr, idx, hidden)
    listed = meta.get("glp.layer_ids_zero_based")
    if isinstance(listed, str) and listed:
        ids = sorted(int(x) for x in listed.split(","))
        if ids != sorted(out):
            raise RuntimeError(f"GB10_ABLATE: {path}: glp.layer_ids_zero_based {ids[:3]}... "
                               f"differs from its direction.<n> tensors {sorted(out)[:3]}...")
    alpha_default = meta.get("glp.alpha_default")
    logger.info("GB10_ABLATE: %s: %s for %s (%s), alpha default %s, rank %s",
                os.path.basename(path), meta.get("general.name", "?"),
                meta.get("glp.base_model", meta.get("controlvector.model_hint", "?")),
                meta.get("glp.method", "?"), alpha_default, meta.get("glp.rank", "?"))
    return out


def load_directions(path: str, hidden: int) -> dict:
    """{layer index: float32 numpy unit vector [hidden]} from a projective GGUF
    control vector (above) or an .npz in either layout:
      * one array per layer, keyed by the layer index ("4", "5", ...); other
        keys (metadata such as a "layers" list) are listed and skipped;
      * one [n, hidden] matrix next to a 1-D "layers" array of n indices."""
    import numpy as np

    if not os.path.isfile(path):
        raise RuntimeError(f"GB10_ABLATE={path}: no such file")
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    with open(path, "rb") as f:
        is_gguf = f.read(4) == b"GGUF"
    if is_gguf:
        out = load_gguf_directions(path, hidden)
        logger.info("GB10_ABLATE: %s (sha256 %s...), %d layers %d..%d", os.path.basename(path),
                    digest[:12], len(out), min(out), max(out))
        return out
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


def find_stream_stack(model):
    """The Qwen3.5 decoder stack inside the target: layers, start/end_layer and
    SGLang's DFlash capture hook, without Qwen4-Exp's hyper-connections."""
    for m in model.modules():
        if (all(hasattr(m, a) for a in ("layers", "start_layer", "end_layer"))
                and hasattr(m, "set_dflash_layers_to_capture") and not hasattr(m, "hc_count")):
            return m
    raise RuntimeError("GB10_ABLATE: no Qwen3.5 decoder stack (layers/start_layer/"
                       "set_dflash_layers_to_capture) in the target model (written for SGLang 0.5.21)")


def project_stream_output(output, stream, d, alpha: float, layer: int) -> None:
    """In place: the layer's pending contribution `output` becomes
    output - alpha d (d . (residual + output)), so residual + output loses alpha
    times its d-component."""
    import torch

    pending = getattr(stream, "pending", None)
    residual = getattr(stream, "residual", None)
    if (pending is None or getattr(pending, "owed", None) is not None
            or getattr(pending, "value", None) is not output or residual is None
            or not isinstance(output, torch.Tensor) or output.shape != residual.shape):
        raise RuntimeError(
            f"GB10_ABLATE: layer {layer}'s output is not a plain pending contribution next to "
            "a written residual (tensor parallel, a fused handoff, or another SGLang than "
            "0.5.21?); unset GB10_ABLATE")
    if output.shape[0] == 0:
        return
    coef = torch.mv(residual, d) + torch.mv(output, d)
    output.addr_(coef, d, alpha=-alpha)


def install_stream(model, cfg: dict) -> list:
    """Qwen3.5: forward hooks on the decoder layers (after the weights load)."""
    import torch

    if cfg["alpha"] == 0:
        logger.info("GB10_ABLATE: alpha 0, the stock model; no hooks")
        return []
    stack = find_stream_stack(model)
    if getattr(stack, "_gb10_ablate_hooks", None):
        return stack._gb10_ablate_hooks
    hidden = int(stack.config.hidden_size)
    dirs = load_directions(cfg["path"], hidden)
    n_layers = len(stack.layers)
    bad = [i for i in dirs if not 0 <= i < n_layers]
    if bad:
        raise RuntimeError(f"GB10_ABLATE: layers {bad} not in the model's 0..{n_layers - 1}")
    if cfg["at"] == "input" and 0 in dirs:
        raise RuntimeError("GB10_ABLATE_AT=input: layer 0's input is the embedding, "
                           "which this hook does not reach; drop layer 0")
    if cfg["alpha"] > 1:
        logger.warning("GB10_ABLATE: alpha %g > 1 reflects the component instead of removing it "
                       "(msuiche's card: 2 refuses harmless prompts, 4 collapses the model)",
                       cfg["alpha"])
    alpha = cfg["alpha"]
    handles = []
    for i in sorted(dirs):
        at = i if cfg["at"] == "output" else i - 1   # the layer whose output gets d_i
        if not stack.start_layer <= at < stack.end_layer:
            continue
        layer = stack.layers[at]
        p = next((q for q in layer.parameters() if q.is_cuda), None)
        device = p.device if p is not None else (
            torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available()
            else torch.device("cpu"))
        d32 = torch.from_numpy(dirs[i]).to(device=device)
        cast = {torch.float32: d32, torch.bfloat16: d32.bfloat16(), torch.float16: d32.half()}

        def hook(_m, _args, kwargs, output, cast=cast, d32=d32, i=i):
            fb = kwargs.get("forward_batch")
            stream = getattr(fb, "residual_stream", None)
            d = cast.get(output.dtype) if isinstance(output, torch.Tensor) else None
            if d is None and isinstance(output, torch.Tensor):
                d = cast.setdefault(output.dtype, d32.to(output.dtype))
            project_stream_output(output, stream, d, alpha, i)
            return None

        handles.append(layer.register_forward_hook(hook, with_kwargs=True))
    cos = adjacent_cosines(dirs)
    stack._gb10_ablate_hooks = handles
    logger.info(
        "GB10_ABLATE: %d layers %d..%d, alpha %g, at layer %s (post-layer residual stream, "
        "Qwen3.5); adjacent directions' cosine %s", len(handles), min(dirs), max(dirs), alpha,
        cfg["at"], f"min {min(cos):.3f} / mean {sum(cos) / len(cos):.3f}" if cos else "n/a")
    return handles


def _wrap_load_weights(cls) -> None:
    orig = cls.load_weights

    def load_weights(self, weights, *args, **kwargs):
        result = orig(self, weights, *args, **kwargs)
        self._gb10_ablate_hooks = install(self, settings())
        return result

    cls.load_weights = load_weights


def _wrap_load_weights_stream(cls) -> None:
    if getattr(cls, "_gb10_ablate_wrapped", False):
        return
    orig = cls.load_weights

    def load_weights(self, weights, *args, **kwargs):
        result = orig(self, weights, *args, **kwargs)
        self._gb10_ablate_hooks = install_stream(self, settings())
        return result

    cls.load_weights = load_weights
    cls._gb10_ablate_wrapped = True


def apply_qwen3_5(mod) -> None:
    found = [name for name in QWEN3_5_CLASSES if getattr(mod, name, None) is not None]
    if not found:
        raise RuntimeError(f"GB10_ABLATE: none of {QWEN3_5_CLASSES} in {QWEN3_5_MODULE} "
                           "(written for SGLang 0.5.21); unset GB10_ABLATE.")
    settings()
    for name in found:
        _wrap_load_weights_stream(getattr(mod, name))
    logger.info("GB10_ABLATE: armed on %s (acts only if this model loads)", " / ".join(found))


def apply_target(mod) -> None:
    cls = getattr(mod, TARGET_CLASS, None)
    if cls is None:
        raise RuntimeError(f"GB10_ABLATE: {TARGET_MODULE}.{TARGET_CLASS} not found "
                           "(written for SGLang 0.5.20); unset GB10_ABLATE.")
    settings()  # fail on a bad setting before the weights load
    _wrap_load_weights(cls)
    logger.info("GB10_ABLATE: armed on %s (acts only if this model loads)", TARGET_CLASS)
