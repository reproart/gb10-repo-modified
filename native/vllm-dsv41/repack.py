#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Adapted from styles01/deepseek-v41-1017-exl3-2x-dgx-sparks repack/repack_ours.py (Apache-2.0,
# data/LICENSE.dgx-spark-deepseek-v41, data/NOTICE.dgx-spark-deepseek-v41; commit in data/SOURCE-COMMIT).
"""Repack the LibertAIDAI DeepSeek-V4.1-Flash-1017 EXL3 2.9bpw pack into the serving tree the
engine reads, as styles01's repack_ours.py does, without touching the two source folders.

Builds a hard-link tree (no copy), appends the Engram tables to a union weight index, rewrites
config.json and exl3_moe_plan.json for the deepseek_v41_exl3 quant method. Standard library only.

  python3 repack.py --pack <MODEL_DIR> --engram <ENGRAM_DIR> --out <SERVE_DIR>

  pack   = LibertAIDAI's 43-shard download (config.json, shards, tokenizer, index)
  engram = the 1017 base's shards 47+48, its index and config.json (the pre-1017 tables do not fit)
  out    = the serving tree: on the same filesystem as both (hard links)

What differs from the original, and why (the result is the same tree):
  * Files that get rewritten (config.json, the index, the plan) are written as new files that
    replace the link, never through it: the original opened them for writing while they were still
    hard links into the pack / Engram folders, which rewrote ENGRAM_DIR's own config.json and index.
    ../tensorfold-dsv41 reads those, and its sha256 check would fail.
  * Directories and dotfiles in the pack folder (hf download's .cache/, .gitattributes) are
    skipped: os.link refuses a directory.
  * chat_template.jinja (and its .bak-*) is not linked: ../tensorfold-dsv41/patch-template.py
    writes one into MODEL_DIR for its own engine; the pack ships none, so the serving tree has none,
    as on styles01's pair.
  * --check verifies an existing tree without writing.
"""
import argparse
import json
import os
import re
import sys

DRAFTER_DENSE_K = {
    "attn.wq_a": 6, "attn.wq_b": 6, "attn.wo_b": 6, "attn.wo_a": 6, "main_proj": 6,
    "ffn.shared_experts.w1": 6, "ffn.shared_experts.w2": 6,
}
DRAFTER_STAGES = 3
DRAFTER_EXPERTS = 128
DRAFTER_DRAFT_K = 4


def link(src, dst):
    if os.path.lexists(dst):
        os.unlink(dst)
    os.link(src, dst)


def write_json(path, obj, **kw):
    """Write a new file and move it over path: a hard link at path is replaced, its target untouched."""
    tmp = path + ".tmp-repack"
    with open(tmp, "w") as f:
        json.dump(obj, f, **kw)
    os.replace(tmp, path)


def linkable(folder, name):
    return (not name.startswith(".") and not name.startswith("chat_template.jinja")
            and os.path.isfile(os.path.join(folder, name)))


def check(out):
    """Exit 0 when out is a finished serving tree: quant method, Engram entries, every index target."""
    try:
        cfg = json.load(open(f"{out}/config.json"))
        wm = json.load(open(f"{out}/model.safetensors.index.json"))["weight_map"]
    except (OSError, ValueError, KeyError) as exc:
        sys.exit(f"not a serving tree: {exc}")
    problems = []
    if cfg.get("quantization_config", {}).get("quant_method") != "deepseek_v41_exl3":
        problems.append("config.json is not rewritten (quant_method)")
    if not any(re.search(r"engram\.embed\.weight$", k) for k in wm):
        problems.append("the index has no Engram entries")
    if not os.path.exists(f"{out}/exl3_moe_plan.json"):
        problems.append("no exl3_moe_plan.json")
    missing = sorted({v for v in wm.values() if not os.path.isfile(os.path.join(out, v))})
    if missing:
        problems.append(f"missing shards: {missing[:3]}{' ...' if len(missing) > 3 else ''}")
    if problems:
        sys.exit("serving tree " + out + ": " + "; ".join(problems))
    print(f"serving tree {out}: OK ({len(set(wm.values()))} shards, Engram indexed)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack")
    ap.add_argument("--engram")
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true", help="verify an existing serving tree, write nothing")
    a = ap.parse_args()
    if a.check:
        return check(a.out)
    if not (a.pack and a.engram):
        ap.error("--pack and --engram are required to build")
    pack, engram, out = a.pack, a.engram, a.out

    idx = json.load(open(f"{pack}/model.safetensors.index.json"))
    wm = idx["weight_map"]
    print(f"pack: {len(wm)} tensors across {len(set(wm.values()))} shards")

    os.makedirs(out, exist_ok=True)
    try:
        dev_out = os.stat(out).st_dev
        for d in (pack, engram):
            if os.stat(d).st_dev != dev_out:
                sys.exit(f"{d} and {out} are on different filesystems: hard links need one (move SERVE_DIR)")
    except OSError as exc:
        sys.exit(str(exc))

    # 1. hard-link the pack files
    for name in set(wm.values()) | {f for f in os.listdir(pack)
                                    if not f.endswith(".safetensors") and linkable(pack, f)}:
        src = os.path.join(pack, name)
        if not os.path.exists(src):
            print(f"WARN missing {name}", file=sys.stderr)
            continue
        link(src, os.path.join(out, name))

    # 2. hard-link the Engram files (its config.json and index replace the pack's links, as in the original)
    eidx = json.load(open(f"{engram}/model.safetensors.index.json"))
    ewm = eidx["weight_map"]
    for name in set(ewm.values()) | {f for f in os.listdir(engram) if f.endswith(".json") and linkable(engram, f)}:
        src = os.path.join(engram, name)
        if not os.path.exists(src):
            # the base index names all 48 shards; only 47 and 48 are downloaded
            continue
        link(src, os.path.join(out, name))

    cfg0 = json.load(open(f"{out}/config.json"))
    nhl = int((cfg0.get("text_config", {}) or cfg0).get("num_hidden_layers") or 40)
    print(f"num_hidden_layers = {nhl}")

    # 3. union weight index: the pack's rows + the Engram entries the loader reads the tables
    #    through (model.-prefixed in the base index; bare layers.N.* in the serving convention)
    eng = {n: ewm[n] for n in ewm if re.search(r"engram\.embed\.(weight|scale)$", n)}
    for entry, shard in eng.items():
        key = re.sub(r"^model\.", "", entry)
        wm[key] = shard
        print(f"index: {key} -> {shard}")
    if not eng:
        sys.exit("FATAL: no engram.embed tensors in the Engram index: is this the 1017 base folder?")
    write_json(f"{out}/model.safetensors.index.json",
               {"metadata": idx.get("metadata", {}) | {"repacked": "1017->serving"}, "weight_map": wm})

    # 4. config.json: quant method + the drafter dense-K map. 'attn.wkv' and
    #    'ffn.shared_experts.w3' are not keys: the loader routes them as shard 1 of fused modules.
    cfg = json.load(open(f"{out}/config.json"))
    q = cfg["quantization_config"]
    q["quant_method"] = "deepseek_v41_exl3"
    q.setdefault("expert_dtype", "fp4")
    ex = q.setdefault("exl3_moe", {})
    ex["plan_file"] = ex.get("plan_file", "exl3_moe_plan.json")
    ex["decode_kernel"] = ex.get("decode_kernel", "coop")
    ex["fused_rows"] = ex.get("fused_rows", 256)
    ex["tp_layout"] = ex.get("tp_layout", "channels")
    ex["head"] = ex.get("head", "exl3")
    ex["head_k"] = 6
    ex["exllamav3_version"] = ex.get("exllamav3_version", "1.6.0")
    d = ex.setdefault("dense", {})
    d.setdefault("layers", [0, nhl - 1])
    d.setdefault("k", {"attn.wo_a": 5, "attn.wo_b": 5, "attn.wq_a": 5, "attn.wq_b": 5,
                       "ffn.shared_experts.w1": 5, "ffn.shared_experts.w2": 5})
    d["drafter"] = dict(DRAFTER_DENSE_K)
    write_json(f"{out}/config.json", cfg, indent=2)
    print("config.json: quant_method=deepseek_v41_exl3 + drafter dense map written")

    # 5. plan: the pack's own (or a uniform fallback) + the drafter section; drafter layers are
    #    absolute indices >= num_layers (= num_hidden_layers + stage)
    if os.path.exists(f"{out}/exl3_moe_plan.json"):
        plan = json.load(open(f"{out}/exl3_moe_plan.json"))
    elif os.path.exists(f"{pack}/exl3_moe_plan.json"):
        plan = json.load(open(f"{pack}/exl3_moe_plan.json"))
    else:
        plan = {"format": "exl3_moe_plan_v1", "num_layers": nhl, "num_experts": 384,
                "hidden_size": 5120, "moe_intermediate_size": 2304,
                "layers": {str(L): {"k": "4" * 384} for L in range(nhl)}}
        print("WARN no plan in the pack; wrote a uniform fallback", file=sys.stderr)
    plan.setdefault("hidden_size", 5120)
    plan.setdefault("num_layers", nhl)
    plan["drafter"] = {"num_experts": DRAFTER_EXPERTS,
                       "layers": {str(nhl + s): {"k": str(DRAFTER_DRAFT_K) * DRAFTER_EXPERTS}
                                  for s in range(DRAFTER_STAGES)}}
    write_json(f"{out}/exl3_moe_plan.json", plan, indent=1)
    print(f"plan: {len(plan['layers'])} backbone layers + drafter {list(plan['drafter']['layers'])}"
          f" ({DRAFTER_EXPERTS} experts @ K{DRAFTER_DRAFT_K})")

    # 6. every index target exists, nonzero
    missing = [n for n in wm if not (os.path.exists(os.path.join(out, wm[n]))
                                     and os.path.getsize(os.path.join(out, wm[n])) > 0)]
    print("VERIFY:", "OK, all index targets present" if not missing else f"MISSING {missing[:5]}", file=sys.stderr)
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
