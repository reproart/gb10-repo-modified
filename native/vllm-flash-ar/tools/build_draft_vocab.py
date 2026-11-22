#!/usr/bin/env python3
"""Build the reduced draft vocabulary for the MTP drafter (qwen38-flash-dgx).

The MTP draft head scores all 248,320 vocabulary rows on every draft step; almost all of them
are never the argmax. This picks the N ids the drafter is allowed to propose:
  1. every special / added token (chat template, tool-call and thinking markers), always,
  2. the first 256 ids (byte fallbacks), always,
  3. with --base, an existing set first: all of it, or with --base-keep K the K of its ids
     most frequent in --base-corpus (e.g. the shipped English/code set, ranked over English
     and code, to make room for another language without growing the head),
  4. every token that appears in --corpus, ranked by count (domain fit),
  5. the remaining slots filled with the lowest ids not yet chosen (BPE merge order is a
     frequency proxy for Qwen tokenizers).
The target model verifies every drafted token, so a token outside this set can never be
emitted *by the draft*; it simply costs a rejection when the target wants it. --holdout texts
(not counted) report the share of their tokens the set covers, for --base and for the result:
the share of positions the draft can propose at all. Output: a sorted int32 .npy of ids.

usage: build_draft_vocab.py <tokenizer_dir> <out.npy> [--n 65536] [--corpus path ...]
         [--base ids.npy [--base-keep K --base-corpus path ...]] [--holdout path ...]

The Russian sets (src/draft_vocab_ru_*.npy) were built this way; README, patch 10.
"""
import argparse, collections, glob, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("tokenizer_dir"); ap.add_argument("out")
ap.add_argument("--n", type=int, default=65536)
ap.add_argument("--corpus", nargs="*", default=[])
ap.add_argument("--base", help="an ids .npy kept before the corpus ranking")
ap.add_argument("--base-keep", type=int, default=0,
                help="keep only this many of --base, the most frequent in --base-corpus (0 = all)")
ap.add_argument("--base-corpus", nargs="*", default=[], help="texts that rank --base for --base-keep")
ap.add_argument("--holdout", nargs="*", default=[], help="texts to report coverage on (not counted)")
a = ap.parse_args()
if a.base_keep and not a.base_corpus:
    ap.error("--base-keep needs --base-corpus to rank the base set")
if a.n % 64:
    print(f"note: --n {a.n} is not a multiple of 64; the int8 / int4 draft head pads it (Marlin needs N % 64 == 0)")

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.tokenizer_dir, trust_remote_code=True)
V = len(tok)
print(f"tokenizer vocab (incl. added): {V}")

def texts_from(path):
    if os.path.isdir(path):
        for f in sorted(glob.glob(os.path.join(path, "**", "*"), recursive=True)):
            if os.path.isfile(f) and os.path.getsize(f) < 20_000_000 and f.split(".")[-1] in ("json", "md", "txt", "py", "log", "jsonl"):
                yield from texts_from(f)
        return
    try:
        raw = open(path, "rb").read().decode("utf-8", errors="ignore")
    except Exception:
        return
    if path.endswith(".json"):
        try:
            def walk(o):
                if isinstance(o, str):
                    yield o
                elif isinstance(o, dict):
                    for v in o.values():
                        yield from walk(v)
                elif isinstance(o, list):
                    for v in o:
                        yield from walk(v)
            yield from walk(json.loads(raw)); return
        except Exception:
            pass
    yield raw

def count(paths, what):
    counts, nchars = collections.Counter(), 0
    for p in paths:
        for t in texts_from(p):
            if len(t) < 20: continue
            nchars += len(t)
            for chunk in (t[i:i+200_000] for i in range(0, len(t), 200_000)):
                counts.update(tok.encode(chunk, add_special_tokens=False))
    print(f"{what}: {nchars/1e6:.1f} M chars, {sum(counts.values())/1e6:.2f} M tokens, {len(counts)} distinct ids")
    return counts

counts = count(a.corpus, "corpus")

keep = set()
# 1. specials / added tokens
specials = set(tok.all_special_ids)
try:
    specials |= set(int(i) for i in tok.added_tokens_decoder.keys())
except Exception:
    pass
keep |= specials
# 2. byte fallbacks
keep |= set(range(min(256, V)))
# 3. the base set, whole or its most frequent part
base = None
if a.base:
    base = [int(i) for i in np.load(a.base)]
    if a.base_keep:
        bc = count(a.base_corpus, "base corpus")
        ranked = sorted(base, key=lambda t: (-bc.get(t, 0), t))
        chosen = set(ranked[:a.base_keep])
        unseen = sum(1 for t in base if bc.get(t, 0) == 0)
        print(f"base: {len(base)} ids from {a.base}, keeping the {a.base_keep} most frequent in the base corpus "
              f"({unseen} of them never appear there)")
    else:
        chosen = set(base)
        print(f"base: {len(base)} ids from {a.base}, all kept")
    keep |= chosen
n_base = len(keep)
# 4. corpus by frequency
for tid, _ in counts.most_common():
    if len(keep) >= a.n: break
    keep.add(int(tid))
n_corpus = len(keep)
# 5. fill with lowest ids
i = 0
while len(keep) < a.n and i < V:
    keep.add(i); i += 1
ids = np.array(sorted(keep), dtype=np.int32)
np.save(a.out, ids)
cov = sum(c for t, c in counts.items() if t in keep) / max(1, sum(counts.values()))
print(f"kept {len(ids)} ids: {len(specials)} special/added, base and fixed up to {n_base}, corpus-ranked "
      f"up to {n_corpus}, filled from id order up to {i}; corpus token coverage {cov*100:.3f}%")
base_set = set(base or ())
for p in a.holdout:
    ids_h = [t for x in texts_from(p) for t in tok.encode(x, add_special_tokens=False)]
    if not ids_h:
        continue
    new = sum(t in keep for t in ids_h) / len(ids_h)
    old = f"; base {sum(t in base_set for t in ids_h) / len(ids_h):.2%}" if base else ""
    print(f"holdout {os.path.basename(p.rstrip('/'))}: {len(ids_h)} tokens, covered {new:.2%}{old}")
print("saved", a.out)
