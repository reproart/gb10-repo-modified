#!/usr/bin/env python3
"""Build the checkpoint view vLLM 0.34 serves (vllm-flash-v034): the recipe's model files and
the lean PLE table as one model directory, without copying a byte of weights.

vLLM 0.34 reads Qwen4Exp's n-gram ("PLE") table straight from the checkpoint's safetensors
files (--engram-config '{"disk_offload": true}': the rows stay in the files, each step's rows
are read on the host). This recipe keeps the table in its own folder (TABLE_DIR, the fp8 rows
of Qwen/Qwen3.8-Flash-Next-FP8), next to a model folder that has none, so the view has:
  * a symlink per model file (the shards, tokenizer, chat template ...) to <model mount>/<f>;
  * a symlink per table shard to <table mount>/<f>;
  * model.safetensors.index.json: the model's weight map plus the table's tensors;
  * config.json: the model's, with text_config.ple_embedding_dtype = "float8_e4m3fn" (with an
    AutoGPTQ quantization config vLLM would take the table for BF16 and refuse the fp8 rows).
The symlinks point at the container's mount points (/model, /ple-table by default), so they
resolve inside the container, not on the host. The table must be lean (only its
ngram_embedding tensors: ../vllm-flash-ar/tools/strip_ple_table.py, ./prepare.sh): vLLM loads
every tensor of every listed file, and the FP8 checkpoint's own shards also carry expert
scales and dense weights of other layers.

Standard library only (runs on the host). Rerunning replaces the view.

usage: make_model_view.py <model_dir> <table_dir> <view_dir> [--model-mount /model] [--table-mount /ple-table]
"""
import argparse
import datetime
import json
import os
import struct
import sys

PLE_DTYPE = "float8_e4m3fn"
STAMP = ".gb10-view.json"


def safetensors_names(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return [k for k in header if k != "__metadata__"], header


def build(model_dir, table_dir, view_dir, model_mount="/model", table_mount="/ple-table"):
    cfg_path = os.path.join(model_dir, "config.json")
    idx_path = os.path.join(model_dir, "model.safetensors.index.json")
    for p in (cfg_path, idx_path):
        if not os.path.isfile(p):
            raise SystemExit(f"make_model_view: no {p}")
    with open(cfg_path) as f:
        config = json.load(f)
    if config.get("model_type") != "qwen4_exp":
        raise SystemExit(f"make_model_view: {cfg_path} has model_type {config.get('model_type')!r}, "
                         "not 'qwen4_exp' (vLLM 0.34's name for Qwen3.8-Flash-Next)")
    with open(idx_path) as f:
        index = json.load(f)
    weight_map = dict(index["weight_map"])

    tables = sorted(f for f in os.listdir(table_dir) if f.endswith(".safetensors"))
    if not tables:
        raise SystemExit(f"make_model_view: no .safetensors in {table_dir}")
    added, foreign = 0, []
    for name in tables:
        tensors, _ = safetensors_names(os.path.join(table_dir, name))
        for t in tensors:
            if "ple_embedding.ngram_embedding." not in t:
                foreign.append(f"{name}: {t}")
                continue
            if t in weight_map:
                raise SystemExit(f"make_model_view: {t} is in both the model and the table")
            weight_map[t] = name
            added += 1
    if foreign:
        raise SystemExit(f"make_model_view: {table_dir} is not a lean table: {len(foreign)} other tensors "
                         f"(e.g. {foreign[0]}); vLLM would load them as model weights. Strip it first: "
                         "./prepare.sh (../vllm-flash-ar/tools/strip_ple_table.py)")
    if not added:
        raise SystemExit(f"make_model_view: no ngram_embedding tensors in {table_dir}")

    model_files = sorted(f for f in os.listdir(model_dir)
                         if os.path.isfile(os.path.join(model_dir, f)) and not f.startswith("."))
    clash = set(model_files) & set(tables)
    if clash:
        raise SystemExit(f"make_model_view: the model and the table both have {sorted(clash)[:3]}")

    os.makedirs(view_dir, exist_ok=True)
    for f in os.listdir(view_dir):          # a previous view: links and the files written below
        p = os.path.join(view_dir, f)
        if os.path.islink(p) or f in ("config.json", "model.safetensors.index.json", STAMP):
            os.remove(p)
        else:
            raise SystemExit(f"make_model_view: {p} is not part of a view; not touching {view_dir}")
    for f in model_files:
        if f in ("config.json", "model.safetensors.index.json"):
            continue
        os.symlink(f"{model_mount}/{f}", os.path.join(view_dir, f))
    for f in tables:
        os.symlink(f"{table_mount}/{f}", os.path.join(view_dir, f))

    text = config.setdefault("text_config", {})
    text["ple_embedding_dtype"] = PLE_DTYPE
    with open(os.path.join(view_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(view_dir, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": index.get("metadata", {}), "weight_map": weight_map}, f, indent=1)
    with open(os.path.join(view_dir, STAMP), "w") as f:
        json.dump({"model_dir": os.path.abspath(model_dir), "table_dir": os.path.abspath(table_dir),
                   "model_mount": model_mount, "table_mount": table_mount,
                   "model_files": len(model_files), "table_files": len(tables), "table_tensors": added,
                   "made": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")},
                  f, indent=1)
    return len(model_files), len(tables), added


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model_dir"); ap.add_argument("table_dir"); ap.add_argument("view_dir")
    ap.add_argument("--model-mount", default="/model"); ap.add_argument("--table-mount", default="/ple-table")
    a = ap.parse_args()
    m, t, n = build(a.model_dir, a.table_dir, a.view_dir, a.model_mount, a.table_mount)
    print(f"view {a.view_dir}: {m} model files -> {a.model_mount}, {t} table shards ({n} tensors) -> "
          f"{a.table_mount}; config.json ple_embedding_dtype={PLE_DTYPE}", file=sys.stderr)
