#!/usr/bin/env python3
"""Keep the GDN gate beta in float32 in SGLang 0.5.20's Triton kernels.

    ~/spark/venv-sglang-0.5.20/bin/python scripts/patch-sglang-gdn-precision.py           # apply
    ~/spark/venv-sglang-0.5.20/bin/python scripts/patch-sglang-gdn-precision.py --check   # report only
    ~/spark/venv-sglang-0.5.20/bin/python scripts/patch-sglang-gdn-precision.py --revert  # undo

Three kernels compute beta = sigmoid(b) in float32, round it to the input
dtype (BF16) and widen it again before it scales the recurrent-state update
of every gated-delta-net (GDN) layer. The fixes upstream keep it in float32
(sgl-project/sglang#38977 for the packed and ReplaySSM decode kernels, #40362
for fused_gdn_gating, which every GDN prefill runs). The edits and file
digests are the ones the infercrane Qwen3.8-27B recipe applies to the same
pinned 0.5.20.

Which of our paths it touches: GDN prefill (fused_gdn_gating) for every
hybrid model here (Qwen3.8-27B, zen6, Meerkat-TRIZ, Ornith, Flash-Next), and
decode without speculative decoding (packed decode) or with ReplaySSM.
DFlash2 / MTP verify steps run fused_sigmoid_gating_delta_rule_update, which
keeps beta in float32 already. A precision fix, not a speed one: compare
answers / HumanEval, not tok/s.

It edits the installed package in the venv, only when each file's sha256 is
the stock 0.5.20 one (or already the patched one); anything else is refused
and nothing is written. Triton keys its cache on the kernel source, so the
next boot recompiles these kernels once. Reinstalling SGLang (pip
--force-reinstall, a new venv) brings the stock files back: run it again.
"""

import argparse
import hashlib
import importlib.util
import pathlib
import sys

PATCHES = {
    "kernels/ops/attention/fla/fused_recurrent.py": (
        "35a928d24bf6cc3ca56d73e4b729ec004ec8a34760e2425e2f04d6ef783db9f8",
        "5fcf3e425d41c5544948b512e1b534897f04bdbac9ac1619ffe224cbb960477a",
        "beta_val = tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)",
        "beta_val = tl.sigmoid(b_val)",
        "sgl-project/sglang#38977",
    ),
    "kernels/ops/attention/fla/fused_recurrent_linear_replayssm.py": (
        "a8824a71ab49fde1f070c325c89603e6198928bdcb2238f2d6e9ef8fb7247f62",
        "a18e01ba74904e8f7d27f1eb2d886af5c139b3de8f2a84290c71b5c041a5aa1a",
        "beta_val = tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)",
        "beta_val = tl.sigmoid(b_val)",
        "sgl-project/sglang#38977",
    ),
    "kernels/ops/attention/fla/fused_gdn_gating.py": (
        "c7736d1e506fb2c3e5c0496a2ed8c23347c2507ebe73c08bc6745517495d0957",
        "7318818b58efcb17803b52f236319eda5cc47b813d64113d1c7b83dd2cac378d",
        "tl.store(beta_output + off, blk_beta_output.to(b.dtype.element_ty), mask=mask)",
        "tl.store(beta_output + off, blk_beta_output, mask=mask)",
        "sgl-project/sglang#40362",
    ),
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sglang_root() -> pathlib.Path:
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit(f"no sglang in this interpreter ({sys.executable}); run it with the "
                         "server venv's python, e.g. ~/spark/venv-sglang-0.5.20/bin/python")
    return pathlib.Path(next(iter(spec.submodule_search_locations)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--check", action="store_true", help="report the state, change nothing")
    ap.add_argument("--revert", action="store_true", help="put the stock lines back")
    args = ap.parse_args()
    root = sglang_root()

    # Check every file first: all of them are changed, or none.
    plan = []
    for rel, (stock, patched, old, new, upstream) in PATCHES.items():
        path = root / rel
        if not path.is_file():
            raise SystemExit(f"{rel}: not in {root}; not SGLang 0.5.20?")
        digest = sha(path.read_bytes())
        if digest == stock:
            state = "stock"
        elif digest == patched:
            state = "patched"
        else:
            raise SystemExit(f"{rel}: unexpected content (sha256 {digest[:16]}...); "
                             "not the SGLang 0.5.20 this script was written for. Nothing changed.")
        plan.append((rel, path, state, stock, patched, old, new, upstream))

    for rel, path, state, stock, patched, old, new, upstream in plan:
        want = "stock" if args.revert else "patched"
        if args.check or state == want:
            print(f"{rel}: {state} ({upstream})")
            continue
        src, dst = (new, old) if args.revert else (old, new)
        # whole-line match: the patched line is a prefix of another line in
        # fused_recurrent.py (".to(tl.float32)" variants), so no substring edits
        lines = path.read_text().splitlines(keepends=True)
        hits = [i for i, ln in enumerate(lines) if ln.strip() == src]
        if len(hits) != 1:
            raise SystemExit(f"{rel}: expected line found {len(hits)} times, not once")
        i = hits[0]
        indent = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
        lines[i] = indent + dst + "\n"
        path.write_text("".join(lines))
        if sha(path.read_bytes()) != (stock if args.revert else patched):
            raise SystemExit(f"{rel}: digest after the edit does not match; check the file")
        print(f"{rel}: {state} -> {want} ({upstream})")
    if not args.check:
        print(f"SGLang at {root}: restart the server; Triton recompiles these kernels once.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
