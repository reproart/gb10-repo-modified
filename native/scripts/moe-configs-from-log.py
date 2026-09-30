#!/usr/bin/env python3
"""Rebuild a Triton MoE config file from a tuner log.

    grep GB10_BEST tune.log | python3 scripts/moe-configs-from-log.py OUT.json

SGLang's tuner, patched by scripts/patch-moe-tuner.py, prints one line per
batch size as soon as it is tuned:

    (BenchmarkWorker pid=...) GB10_BEST 16 {"BLOCK_SIZE_M": 16, ...} 123.45 us

This turns those lines into the {batch size: config} JSON SGLang reads, so
a run stopped before the last batch size (the tuner writes its file only at
the very end) still yields the sizes it finished. SGLang uses the nearest
tuned batch size for the others. A later line for the same size wins.
"""

import json
import re
import sys

LINE = re.compile(r"GB10_BEST (\d+) (\{.*\}) ([0-9.]+) us")


def parse(lines) -> dict:
    configs = {}
    for line in lines:
        m = LINE.search(line)
        if m:
            configs[int(m.group(1))] = json.loads(m.group(2))
    return dict(sorted(configs.items()))


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__.strip().splitlines()[2], file=sys.stderr)
        return 2
    configs = parse(sys.stdin)
    if not configs:
        print("no GB10_BEST lines on stdin", file=sys.stderr)
        return 1
    with open(sys.argv[1], "w") as f:
        json.dump({str(k): v for k, v in configs.items()}, f, indent=4)
        f.write("\n")
    print(f"{sys.argv[1]}: batch sizes {', '.join(map(str, configs))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
