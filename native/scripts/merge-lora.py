#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter into a BF16 safetensors checkpoint, shard by shard.

    python3 scripts/merge-lora.py BASE_DIR ADAPTER_DIR OUT_DIR [--chat-template adapter]

e.g. Meerkat-AI/Meerkat-TRIZ-v1-Qwen3.8-27B into Qwen/Qwen3.8-27B:

    python3 scripts/merge-lora.py /models/Qwen3.8-27B /models/Meerkat-TRIZ-v1-Qwen3.8-27B \\
        /models/Meerkat-TRIZ-v1-Qwen3.8-27B-merged

For every adapter pair the base weight gets W += (alpha / r) * B @ A
(alpha / sqrt(r) with rslora), computed in float32 and stored back in the
base weight's dtype; everything else is copied as is. One shard is in memory
at a time (~5 GB for the 27B), so neither transformers nor peft is needed and
the full model never has to fit. The merge runs on the GPU when torch sees
one (stop the server first: the box has one pool of memory), else on the CPU.

Strict: every lora_A / lora_B pair must find exactly one BF16/FP16/FP32 base
weight of the right shape, or nothing is written. The base has to be the
unquantized checkpoint the adapter was trained on; merging into FP8 or NVFP4
weights would quantize twice. Quantize the merged result instead (e.g.
SGLang --quantization fp8 at load).

Config, tokenizer and the other small files come from the base. The adapter's
chat template is used only with --chat-template adapter; the script says
whether the two differ.
"""

import argparse
import filecmp
import json
import math
import re
import shutil
import sys
from pathlib import Path

PAIR = re.compile(r"^(?:base_model\.model\.)?(?P<module>.+)\.lora_(?P<ab>[AB])(?:\.[^.]+)?\.weight$")


def read_adapter_config(adapter: Path) -> dict:
    cfg = json.loads((adapter / "adapter_config.json").read_text())
    if cfg.get("peft_type") != "LORA":
        raise SystemExit(f"not a LoRA adapter: peft_type={cfg.get('peft_type')}")
    for key in ("rank_pattern", "alpha_pattern"):
        if cfg.get(key):
            raise SystemExit(f"{key} is set; per-module ranks are not supported here")
    for key in ("use_dora", "fan_in_fan_out", "modules_to_save", "trainable_token_indices"):
        if cfg.get(key):
            raise SystemExit(f"{key} is set; not supported here")
    if cfg.get("bias", "none") != "none" or cfg.get("lora_bias"):
        raise SystemExit("adapter trains biases; not supported here")
    return cfg


def scaling(cfg: dict) -> float:
    r, alpha = cfg["r"], cfg["lora_alpha"]
    return alpha / math.sqrt(r) if cfg.get("use_rslora") else alpha / r


def load_pairs(adapter: Path) -> dict:
    """{module name: {"A": tensor, "B": tensor}} from adapter_model.safetensors."""
    from safetensors.torch import load_file

    pairs = {}
    for key, t in load_file(str(adapter / "adapter_model.safetensors")).items():
        m = PAIR.match(key)
        if not m:
            raise SystemExit(f"unexpected tensor in the adapter: {key}")
        pairs.setdefault(m["module"], {})[m["ab"]] = t
    broken = [k for k, v in pairs.items() if set(v) != {"A", "B"}]
    if broken:
        raise SystemExit(f"{len(broken)} modules without both lora_A and lora_B, e.g. {broken[0]}")
    return pairs


def base_shards(base: Path) -> list:
    index = base / "model.safetensors.index.json"
    if index.exists():
        return sorted(set(json.loads(index.read_text())["weight_map"].values()))
    single = base / "model.safetensors"
    if single.exists():
        return [single.name]
    raise SystemExit(f"no model.safetensors(.index.json) in {base}")


# Where a text-only training wrapper's names live in a multimodal checkpoint:
# Qwen3_5ForCausalLM saves model.layers.N..., Qwen3_5ForConditionalGeneration
# keeps the same weights under model.language_model.layers.N...
PREFIX_MAP = (("model.", "model.language_model."),)
# Parts of a checkpoint a language-model adapter never trains: the MTP head
# (same layer names as layer 0..) and the vision tower.
NOT_TRAINED = re.compile(r"^(mtp\.|model\.visual\.|visual\.|model\.vision_tower\.)")


def resolve(pairs: dict, base_keys: set) -> dict:
    """Adapter module -> base weight key. Tried in order: the name as is
    (+ ".weight"); the known prefix rewrites (PREFIX_MAP); a unique suffix
    match among the language-model weights (MTP and vision excluded, so
    model.layers.0.mlp.down_proj cannot land on mtp.layers.0.mlp.down_proj)."""
    lm_keys = {k for k in base_keys if not NOT_TRAINED.match(k)}
    by_suffix = {}
    for k in lm_keys:
        parts = k.split(".")
        for i in range(len(parts)):
            by_suffix.setdefault(".".join(parts[i:]), []).append(k)
    out, how = {}, {}
    for module in pairs:
        key = module + ".weight"
        found, rule = None, None
        if key in base_keys:
            found, rule = key, "as is"
        for old, new in PREFIX_MAP:
            if found is None and key.startswith(old) and new + key[len(old):] in lm_keys:
                found, rule = new + key[len(old):], f"{old}* -> {new}*"
        if found is None:
            parts = key.split(".")
            cands = []
            for i in range(len(parts) - 1):
                cands = by_suffix.get(".".join(parts[i:]), [])
                if cands:
                    break
            if len(cands) != 1:
                raise SystemExit(f"adapter module {module}: {len(cands)} matching base weights "
                                 f"({cands[:3]}); refusing to guess")
            found, rule = cands[0], "suffix"
        out[module] = found
        how[rule] = how.get(rule, 0) + 1
    dup = len(out) - len(set(out.values()))
    if dup:
        raise SystemExit(f"{dup} base weights claimed by two adapter modules")
    print("name mapping: " + ", ".join(f"{n} {r}" for r, n in how.items()))
    return out


def merge_into(w, a, b, s, device):
    import torch

    if w.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise SystemExit(f"base weight dtype {w.dtype}: merge into the unquantized checkpoint")
    if w.dim() != 2 or b.shape[0] != w.shape[0] or a.shape[1] != w.shape[1] or a.shape[0] != b.shape[1]:
        raise SystemExit(f"shape mismatch: W {tuple(w.shape)}, A {tuple(a.shape)}, B {tuple(b.shape)}")
    wf = w.to(device, torch.float32)
    wf.addmm_(b.to(device, torch.float32), a.to(device, torch.float32), alpha=s)
    return wf.to(w.dtype).cpu()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("base", type=Path)
    ap.add_argument("adapter", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--chat-template", choices=("base", "adapter"), default="base")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    args = ap.parse_args()

    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} exists and is not empty")
    cfg = read_adapter_config(args.adapter)
    s = scaling(cfg)
    pairs = load_pairs(args.adapter)
    shards = base_shards(args.base)
    keys_of = {}
    for shard in shards:
        with safe_open(str(args.base / shard), "pt") as f:
            keys_of[shard] = set(f.keys())
    target = resolve(pairs, set().union(*keys_of.values()))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"adapter: {len(pairs)} LoRA pairs, r={cfg['r']}, alpha={cfg['lora_alpha']}, "
          f"scaling {s:g}; base: {len(shards)} shards; merging on {device}")

    args.out.mkdir(parents=True, exist_ok=True)
    for p in args.base.iterdir():
        if p.is_file() and not p.name.endswith(".safetensors"):
            shutil.copy2(p, args.out / p.name)
    tmpl = "chat_template.jinja"
    if (args.adapter / tmpl).exists():
        same = (args.base / tmpl).exists() and filecmp.cmp(args.base / tmpl, args.adapter / tmpl, False)
        print(f"chat template: the adapter's is {'the same as' if same else 'DIFFERENT from'} "
              f"the base's; using the {args.chat_template}'s")
        if args.chat_template == "adapter":
            shutil.copy2(args.adapter / tmpl, args.out / tmpl)

    done = 0
    for shard in shards:
        tensors = load_file(str(args.base / shard))
        mine = [(m, k) for m, k in target.items() if k in keys_of[shard]]
        for module, key in mine:
            tensors[key] = merge_into(tensors[key], pairs[module]["A"], pairs[module]["B"], s, device)
        save_file(tensors, str(args.out / shard), metadata={"format": "pt"})
        done += len(mine)
        print(f"  {shard}: {len(mine)} weights merged ({done}/{len(target)})", flush=True)
        del tensors
    if done != len(pairs):
        raise SystemExit(f"merged {done} of {len(pairs)} pairs")
    (args.out / "merged_from.json").write_text(json.dumps({
        "base": str(args.base), "adapter": str(args.adapter), "pairs": len(pairs),
        "r": cfg["r"], "lora_alpha": cfg["lora_alpha"], "scaling": s,
        "chat_template": args.chat_template}, indent=2) + "\n")
    print(f"done: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
