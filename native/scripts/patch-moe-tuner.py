#!/usr/bin/env python3
"""Make SGLang's Triton MoE tuner handle FP8 per-channel checkpoints.

    python3 scripts/patch-moe-tuner.py ~/sglang-src/benchmark/kernels/fused_moe_triton

SGLang v0.5.20's benchmark/kernels/fused_moe_triton tuner, run on a
compressed-tensors FP8 checkpoint with per-channel weights (e.g.
ornith-ai/Ornith-1.5-35B-A3B-FP8: `strategy: channel`, dynamic per-token
activations), has two faults:

  * common_utils.get_model_config reads any `config_groups` as group
    quantization and sets block_shape = [0, group_size]; per-channel has no
    group_size, so it becomes [0, None] and the search-space filter fails
    ("unsupported operand type(s) for %: 'NoneType' and 'int").
  * benchmark_config builds per-channel scales and dynamic activations only
    for int8 w8a8; FP8 with --per-channel-quant got per-tensor scales and a
    static activation scale, i.e. not the kernel path the server runs.

Both edits are exact-string replacements; the script refuses to touch a file
that does not contain the expected text, and running it twice is a no-op.
"""

import sys
from pathlib import Path

EDITS = {
    "common_utils.py": (
        """        group_size = weights_config.get("group_size")
        block_shape = [0, group_size]
        assert len(block_shape) == 2""",
        """        group_size = weights_config.get("group_size")
        # per-channel / per-tensor schemes (FP8 `strategy: channel`) have no groups
        if group_size is not None:
            block_shape = [0, group_size]
            assert len(block_shape) == 2""",
    ),
    "tuning_fused_moe_triton.py": (
        """        if use_int8_w8a8 and block_shape is None:""",
        """        # per-channel weight scales and dynamic per-token activations, the
        # path the server runs (FP8 per-channel too, not only int8)
        if (use_int8_w8a8 or per_channel_quant) and block_shape is None:""",
    ),
}


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__.strip().splitlines()[2], file=sys.stderr)
        return 2
    root = Path(sys.argv[1]).expanduser()
    for name, (old, new) in EDITS.items():
        path = root / name
        text = path.read_text()
        if new in text:
            print(f"{name}: already patched")
            continue
        if text.count(old) != 1:
            print(f"{name}: expected text not found once; not the v0.5.20 tuner?", file=sys.stderr)
            return 1
        path.write_text(text.replace(old, new))
        print(f"{name}: patched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
