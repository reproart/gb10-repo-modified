#!/usr/bin/env python3
"""Build-time patch (qwen38-flash-dgx): refusal-direction ablation at inference (VLLM_ABLATE).

Inserts a projection after every writer of the hyper-connection streams in vLLM's
models/qwen3_8_flash_next/nvidia/model.py (the token embedding before it is repeated into the
streams, the PLE addition, attention's / GDN's output, the MLP's output, deepstack embeddings)
and after the MTP draft's fused input in mtp.py, each guarded by a per-module flag that is
False unless VLLM_ABLATE is set at runtime. With it set, gb10_abliterate (copied next to vLLM
by the Dockerfile) loads the direction(s), registers them as a buffer in the modules that
project, and sets their flags; src/gb10_abliterate.py explains the math and the knobs.

Without VLLM_ABLATE the inserted lines are `if self._gb10_abl_on:` on a class attribute that
is False: torch.compile specializes them away and the model runs as before (the PLE addition
is split over two lines, the same operations).

usage: patch_abliterate.py <path to .../nvidia/model.py> <path to .../nvidia/mtp.py>
Each insertion point must match exactly once, or the build stops (a different vLLM).
"""
import ast
import sys

MARK = "qwen38-flash-dgx: refusal-direction ablation"

MODEL_EDITS = [
    (  # the embedding writes all four streams
        "            hidden_states = hidden_states.repeat(1, self.config.hc_count)\n",
        "            if self._gb10_abl_on:  # qwen38-flash-dgx ablation: the embedding (src/patch_abliterate.py)\n"
        "                hidden_states = _gb10_abl_project(hidden_states, self._gb10_abl_u)\n"
        "            hidden_states = hidden_states.repeat(1, self.config.hc_count)\n",
    ),
    (  # deepstack embeddings (images) are added to every stream after their layer
        "                deepstack_embed = deepstack_input_embeds[\n"
        "                    f\"deepstack_input_embeds_{layer_idx}\"\n"
        "                ]\n",
        "                deepstack_embed = deepstack_input_embeds[\n"
        "                    f\"deepstack_input_embeds_{layer_idx}\"\n"
        "                ]\n"
        "                if layer._gb10_abl_on:  # qwen38-flash-dgx ablation: deepstack\n"
        "                    deepstack_embed = _gb10_abl_project(deepstack_embed, layer._gb10_abl_u)\n",
    ),
    (  # PLE adds [T, 4 x H] to the streams directly
        "            hidden_states = hidden_states + self.ple(\n"
        "                hidden_states,\n"
        "                input_ids,\n"
        "                query_start_loc,\n"
        "                ngram_context,\n"
        "            )\n",
        "            ple_out = self.ple(\n"
        "                hidden_states,\n"
        "                input_ids,\n"
        "                query_start_loc,\n"
        "                ngram_context,\n"
        "            )\n"
        "            if self._gb10_abl_on:  # qwen38-flash-dgx ablation: PLE\n"
        "                ple_out = _gb10_abl_project(ple_out, self._gb10_abl_u)\n"
        "            hidden_states = hidden_states + ple_out\n",
    ),
    (  # attention / GDN output, combined into the streams by the MLP's HC module
        "        mlp_hc = self.mlp_hyper_connection\n"
        "        hidden_states, block_input, injection = mlp_hc.combine_and_mix(\n"
        "            hidden_states, attn_out, injection\n"
        "        )\n",
        "        if self._gb10_abl_on:  # qwen38-flash-dgx ablation: attention / GDN output\n"
        "            attn_out = _gb10_abl_project(attn_out, self._gb10_abl_u)\n"
        "        mlp_hc = self.mlp_hyper_connection\n"
        "        hidden_states, block_input, injection = mlp_hc.combine_and_mix(\n"
        "            hidden_states, attn_out, injection\n"
        "        )\n",
    ),
    (  # the MLP / MoE output, combined in the next layer (delayed combine)
        "        mlp_out = self.mlp(block_input)\n"
        "        return hidden_states, mlp_out, injection\n",
        "        mlp_out = self.mlp(block_input)\n"
        "        if self._gb10_abl_on:  # qwen38-flash-dgx ablation: MLP / MoE output\n"
        "            mlp_out = _gb10_abl_project(mlp_out, self._gb10_abl_u)\n"
        "        return hidden_states, mlp_out, injection\n",
    ),
]

MODEL_HOOK = '''

# --- qwen38-flash-dgx: refusal-direction ablation (VLLM_ABLATE=<vectors>; src/patch_abliterate.py)
# The forward code above checks these flags; they stay False unless VLLM_ABLATE is set, when
# gb10_abliterate wraps both classes' __init__ to attach the direction to each instance.
Qwen3_8FlashNextDecoderLayer._gb10_abl_on = False
Qwen3_8FlashNextModel._gb10_abl_on = False
import os as _gb10_abl_os

if _gb10_abl_os.environ.get("VLLM_ABLATE"):
    import gb10_abliterate as _gb10_abl

    _gb10_abl_project = _gb10_abl.project
    _gb10_abl.install(globals())
'''

MTP_EDITS = [
    (  # fc_embedding + fc_hidden write the draft layer's streams
        "        current_step_idx = spec_step_idx % self.num_mtp_layers\n"
        "        layer = self.layers[current_step_idx]\n",
        "        current_step_idx = spec_step_idx % self.num_mtp_layers\n"
        "        layer = self.layers[current_step_idx]\n"
        "        if layer._gb10_abl_on:  # qwen38-flash-dgx ablation: the draft's fused input\n"
        "            hidden_states = _gb10_abl_project(hidden_states, layer._gb10_abl_u)\n",
    ),
]

MTP_HOOK = '''

# --- qwen38-flash-dgx: refusal-direction ablation in the MTP draft (VLLM_ABLATE; src/patch_abliterate.py)
# The draft layer is model.py's decoder layer: its flag and buffer come from there
# (VLLM_ABLATE_MTP=0 leaves it False).
import os as _gb10_abl_os

if _gb10_abl_os.environ.get("VLLM_ABLATE"):
    from gb10_abliterate import project as _gb10_abl_project
'''


def patch(path: str, edits, hook: str) -> str:
    with open(path) as f:
        src = f.read()
    if MARK in src:
        return "already installed"
    for old, new in edits:
        n = src.count(old)
        if n != 1:
            first = old.strip().splitlines()[0]
            raise SystemExit(f"patch_abliterate: {path}: {n} matches for {first!r} (expected 1): "
                             "not the file this patch was written for")
        src = src.replace(old, new)
    out = src.rstrip("\n") + hook
    ast.parse(out)
    with open(path, "w") as f:
        f.write(out)
    return f"INSTALLED ({len(edits)} insertion point{'s' if len(edits) > 1 else ''})"


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__.strip().splitlines()[-2])
    model_py, mtp_py = sys.argv[1], sys.argv[2]
    print("  ablation hook", patch(model_py, MODEL_EDITS, MODEL_HOOK), "in", model_py)
    print("  ablation hook", patch(mtp_py, MTP_EDITS, MTP_HOOK), "in", mtp_py,
          "(inert unless VLLM_ABLATE is set)")
