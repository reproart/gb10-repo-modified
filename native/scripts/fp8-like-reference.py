#!/usr/bin/env python3
"""Quantize a BF16 checkpoint to block FP8 exactly as a reference FP8 checkpoint is laid out.

    # 1. check the method: quantize the reference's own BF16 source and compare
    python3 scripts/fp8-like-reference.py check /models/Qwen3.8-27B /models/Qwen3.8-27B-FP8
    # 2. convert, e.g. the merged Meerkat-TRIZ checkpoint
    python3 scripts/fp8-like-reference.py convert /models/Meerkat-TRIZ-v1-Qwen3.8-27B-merged \\
        /models/Qwen3.8-27B-FP8 /models/Meerkat-TRIZ-v1-Qwen3.8-27B-FP8

Made for a finetune of a model that has an official FP8 release (Qwen's
Qwen3.8-27B-FP8: FP8 E4M3 weights in 128 x 128 blocks, one scale per block
stored as `<name>.weight_scale_inv`, activations quantized dynamically at
run time). The reference decides everything: which tensors are FP8 (those
it stores as F8_E4M3; the rest stay as the source has them), the block size
(its config's quantization_config.weight_block_size), the scale dtype, and
the quantization_config written into the new config.json. The result loads
like the reference: same SGLang path, same kernels, no --quantization flag.

Per block: scale = max|w| / 448, q = round_to_nearest(w / scale) as E4M3,
stored scale_inv = scale (the dequantization multiplier, the name's
convention). `check` measures how close that comes to the reference on its
own BF16 source: identical FP8 bytes and scales mean the converted finetune
is quantized the way the official checkpoint was.

Strict: every reference FP8 tensor must exist in the source with the same
shape, and the source may not have weights the reference lacks (or the
reverse), unless --allow-missing / --allow-extra. One source shard in
memory at a time (~5 GB), on the GPU when torch sees one (stop the server
first: one pool of memory) or the CPU. Run it with the server venv's python
(torch, safetensors). Output: the source's shard names, a fresh
model.safetensors.index.json, the source's small files (tokenizer,
templates), config.json with the reference's quantization_config, and
fp8_from.json recording where it came from.
"""

import argparse
import json
import math
import random
import shutil
import sys
from pathlib import Path

FP8_MAX = 448.0
SCALE_SUFFIX = "_scale_inv"


def shard_list(d: Path) -> list:
    idx = d / "model.safetensors.index.json"
    if idx.exists():
        return sorted(set(json.loads(idx.read_text())["weight_map"].values()))
    if (d / "model.safetensors").exists():
        return ["model.safetensors"]
    raise SystemExit(f"no model.safetensors(.index.json) in {d}")


def inventory(d: Path) -> dict:
    """{tensor name: (dtype string, shape tuple, shard)} from the headers only."""
    from safetensors import safe_open

    out = {}
    for shard in shard_list(d):
        with safe_open(str(d / shard), "pt") as f:
            for k in f.keys():
                s = f.get_slice(k)
                out[k] = (s.get_dtype(), tuple(s.get_shape()), shard)
    return out


def ref_layout(ref: Path):
    """(fp8 weight names, block size, scale dtype string, quantization_config)."""
    cfg = json.loads((ref / "config.json").read_text())
    qc = cfg.get("quantization_config") or cfg.get("text_config", {}).get("quantization_config")
    if not qc:
        raise SystemExit(f"{ref}/config.json has no quantization_config: not an FP8 checkpoint?")
    block = qc.get("weight_block_size")
    if not block or len(block) != 2:
        raise SystemExit(f"reference quantization_config has no 2-D weight_block_size ({qc}); "
                         "only block FP8 is supported")
    inv = inventory(ref)
    fp8 = {k for k, (dt, _, _) in inv.items() if dt == "F8_E4M3"}
    if not fp8:
        raise SystemExit(f"no F8_E4M3 tensors in {ref}")
    scale_dtypes = set()
    for k in fp8:
        sk = k + SCALE_SUFFIX
        if sk not in inv:
            raise SystemExit(f"reference FP8 tensor {k} has no {sk}")
        n, kk = inv[k][1]
        want = (math.ceil(n / block[0]), math.ceil(kk / block[1]))
        if inv[sk][1] != want:
            raise SystemExit(f"{sk} shape {inv[sk][1]}, expected {want} for {block} blocks")
        scale_dtypes.add(inv[sk][0])
    if len(scale_dtypes) != 1:
        raise SystemExit(f"mixed scale dtypes in the reference: {scale_dtypes}")
    return inv, fp8, tuple(block), scale_dtypes.pop(), qc


TORCH_DTYPE = {"F32": "float32", "BF16": "bfloat16", "F16": "float16"}


def quantize(w, block, scale_dtype, device):
    """BF16/FP16/FP32 [N, K] -> (FP8 E4M3 [N, K], scale [ceil(N/bn), ceil(K/bk)])."""
    import torch

    bn, bk = block
    n, k = w.shape
    nb, kb = math.ceil(n / bn), math.ceil(k / bk)
    x = w.to(device=device, dtype=torch.float32)
    x = torch.nn.functional.pad(x, (0, kb * bk - k, 0, nb * bn - n))
    x = x.view(nb, bn, kb, bk)
    amax = x.abs().amax(dim=(1, 3))
    scale = (amax / FP8_MAX).clamp_min(1e-12)
    q = (x / scale[:, None, :, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    q = q.view(nb * bn, kb * bk)[:n, :k].contiguous()
    return q.cpu(), scale.to(getattr(torch, TORCH_DTYPE[scale_dtype])).cpu()


def dequant(q, scale, block):

    bn, bk = block
    s = scale.float().repeat_interleave(bn, 0)[: q.shape[0]].repeat_interleave(bk, 1)[:, : q.shape[1]]
    return q.float() * s


def device_of(arg):
    import torch

    return arg or ("cuda" if torch.cuda.is_available() else "cpu")


def check_names(src_inv, ref_inv, fp8, allow_missing, allow_extra):
    ref_plain = {k for k in ref_inv if not k.endswith(SCALE_SUFFIX)}
    missing = sorted(ref_plain - set(src_inv))
    extra = sorted(set(src_inv) - ref_plain)
    shape_bad = [k for k in fp8 if k in src_inv and src_inv[k][1] != ref_inv[k][1]]
    if shape_bad:
        raise SystemExit(f"{len(shape_bad)} FP8 tensors differ in shape, e.g. {shape_bad[0]}: "
                         f"source {src_inv[shape_bad[0]][1]}, reference {ref_inv[shape_bad[0]][1]}")
    bad_dtype = [k for k in fp8 if k in src_inv and src_inv[k][0] not in TORCH_DTYPE]
    if bad_dtype:
        raise SystemExit(f"source tensor {bad_dtype[0]} is {src_inv[bad_dtype[0]][0]}: "
                         "convert from the unquantized (BF16) checkpoint")
    if missing and not allow_missing:
        raise SystemExit(f"{len(missing)} reference tensors not in the source, e.g. {missing[:3]} "
                         "(--allow-missing to go on)")
    if extra and not allow_extra:
        raise SystemExit(f"{len(extra)} source tensors not in the reference, e.g. {extra[:3]} "
                         "(--allow-extra keeps them as they are)")
    return missing, extra


def cmd_check(args) -> int:
    import torch
    from safetensors import safe_open

    ref_inv, fp8, block, scale_dtype, _ = ref_layout(args.ref)
    src_inv = inventory(args.src)
    check_names(src_inv, ref_inv, fp8, True, True)
    names = sorted(k for k in fp8 if k in src_inv)
    random.seed(0)
    sample = random.sample(names, min(args.samples, len(names)))
    device = device_of(args.device)
    print(f"reference: {len(fp8)} FP8 tensors, blocks {block}, scales {scale_dtype}; "
          f"checking {len(sample)} on {device}")
    worst_bytes, worst_scale = 1.0, 0.0
    for k in sample:
        with safe_open(str(args.src / src_inv[k][2]), "pt") as f:
            w = f.get_tensor(k)
        with safe_open(str(args.ref / ref_inv[k][2]), "pt") as f:
            rq = f.get_tensor(k)
        with safe_open(str(args.ref / ref_inv[k + SCALE_SUFFIX][2]), "pt") as f:
            rs = f.get_tensor(k + SCALE_SUFFIX)
        q, s = quantize(w, block, scale_dtype, device)
        same = (q.view(torch.uint8) == rq.view(torch.uint8)).float().mean().item()
        srel = ((s.float() - rs.float()).abs() / rs.float().abs().clamp_min(1e-20)).max().item()
        werr = ((dequant(rq, rs, block) - w.float()).norm() / w.float().norm()).item()
        worst_bytes, worst_scale = min(worst_bytes, same), max(worst_scale, srel)
        print(f"  {k}: FP8 bytes identical {same:.4%}, scale max rel diff {srel:.2e}, "
              f"reference rel error {werr:.2e}")
    print(f"worst: {worst_bytes:.4%} identical bytes, scale diff {worst_scale:.2e}")
    if worst_bytes > 0.999 and worst_scale < 1e-3:
        print("same method as the reference: a conversion with `convert` matches its layout and rounding")
    else:
        print("NOT the reference's exact method (different rounding or scale rule): a converted "
              "checkpoint still loads the same way, but is not quantized byte-for-byte alike")
    return 0


def cmd_convert(args) -> int:
    from safetensors.torch import load_file, save_file

    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} exists and is not empty")
    ref_inv, fp8, block, scale_dtype, qc = ref_layout(args.ref)
    src_inv = inventory(args.src)
    missing, extra = check_names(src_inv, ref_inv, fp8, args.allow_missing, args.allow_extra)
    device = device_of(args.device)
    todo = sorted(k for k in fp8 if k in src_inv)
    print(f"{args.src} -> {args.out}: {len(todo)} tensors to FP8 ({block[0]}x{block[1]} blocks, "
          f"{scale_dtype} scales), {len(src_inv) - len(todo)} kept; on {device}")
    if missing:
        print(f"  {len(missing)} reference tensors absent from the source (kept absent)")
    if extra:
        print(f"  {len(extra)} source tensors absent from the reference (kept as they are)")

    args.out.mkdir(parents=True, exist_ok=True)
    weight_map, total, done, worst = {}, 0, 0, 0.0
    for shard in shard_list(args.src):
        tensors = load_file(str(args.src / shard))
        out = {}
        for k, t in tensors.items():
            if k in fp8:
                q, s = quantize(t, block, scale_dtype, device)
                err = ((dequant(q, s, block) - t.float()).norm() / t.float().norm().clamp_min(1e-20)).item()
                worst = max(worst, err)
                out[k], out[k + SCALE_SUFFIX] = q, s
                done += 1
            else:
                out[k] = t
        save_file(out, str(args.out / shard), metadata={"format": "pt"})
        for k, t in out.items():
            weight_map[k] = shard
            total += t.numel() * t.element_size()
        print(f"  {shard}: {sum(1 for k in tensors if k in fp8)} to FP8 ({done}/{len(todo)})", flush=True)
        del tensors, out
    if done != len(todo):
        raise SystemExit(f"quantized {done} of {len(todo)} tensors")

    (args.out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))},
        indent=2) + "\n")
    for p in args.src.iterdir():
        if p.is_file() and not p.name.endswith(".safetensors") and p.name not in (
                "model.safetensors.index.json", "config.json"):
            shutil.copy2(p, args.out / p.name)
    # where the reference keeps it: top level, or inside text_config
    cfg = json.loads((args.src / "config.json").read_text())
    ref_cfg = json.loads((args.ref / "config.json").read_text())
    if "quantization_config" in ref_cfg:
        cfg["quantization_config"] = qc
    else:
        cfg.setdefault("text_config", {})["quantization_config"] = qc
    (args.out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (args.out / "fp8_from.json").write_text(json.dumps({
        "source": str(args.src), "reference": str(args.ref), "fp8_tensors": done,
        "block": list(block), "scale_dtype": scale_dtype,
        "worst_relative_error": worst}, indent=2) + "\n")
    print(f"done: {args.out} ({total / 2**30:.1f} GiB); worst per-tensor relative error {worst:.2e}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="quantize SRC tensors and compare with the reference")
    c.add_argument("src", type=Path)
    c.add_argument("ref", type=Path)
    c.add_argument("--samples", type=int, default=12)
    c.add_argument("--device", default=None)
    v = sub.add_parser("convert", help="write SRC as an FP8 checkpoint laid out like REF")
    v.add_argument("src", type=Path)
    v.add_argument("ref", type=Path)
    v.add_argument("out", type=Path)
    v.add_argument("--allow-missing", action="store_true")
    v.add_argument("--allow-extra", action="store_true")
    v.add_argument("--device", default=None)
    args = ap.parse_args()
    return cmd_check(args) if args.cmd == "check" else cmd_convert(args)


if __name__ == "__main__":
    sys.exit(main())
