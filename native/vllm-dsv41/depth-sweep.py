#!/usr/bin/env python3
"""Re-measure the DSpark round-cost table (ROUND_MS) on this boot, as styles01's depth_sweep.sh does.

Boot with DEPTH_PROBE=1 ./serve.sh: rank 0 then reads its depth policy from CACHE_DIR/dspark_depth
every 64 verify steps. For each fixed depth this writes the policy there, sends a long request so the
engine re-reads it, then times three greedy streamed replies (code, prose, structured): each streamed
chunk is one verify round, so ms per round = decode time / chunks and tokens per round = tokens /
chunks. The medians over the three prompts by depth are the table; the last pass runs the adaptive
policy with it ("conf:<table>") to show what it gives. Then set it and boot normally:

    ROUND_MS=<printed table> ./serve.sh        (or in config.local.sh)

    python3 depth-sweep.py                     depths 0..DRAFT_TOKENS, 2 reps each
    python3 depth-sweep.py --depths 0 1 2 3 --reps 3
The engine's own log line every 200 steps (VLLM_DSPARK_DEPTH_LOG) shows the policy it runs.
"""
import argparse
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "bench"))
os.environ.setdefault("GB10_RUN_DIR", "")

PROMPTS = {
    "code": "Write a complete Python implementation of an LRUCache class with get and put in O(1), "
            "using a dict and a doubly linked list. Include docstrings.",
    "prose": "Write a calm, descriptive essay of about 500 words about a walk along a cold northern "
             "coastline at dawn: the light, the sounds, the people met on the way.",
    "structured": "Return a JSON array of 25 objects describing fictional employees, each with the keys "
                  "id, name, department, title, start_date (YYYY-MM-DD), salary and manager_id. "
                  "Output only the JSON.",
}


def config_value(name):
    out = subprocess.run(["bash", "-c", f'. "{HERE}/config.sh" >/dev/null && printf %s "${name}"'],
                         stdout=subprocess.PIPE, check=True)
    return out.stdout.decode()


def write_policy(path, policy):
    try:
        with open(path, "w") as f:
            f.write(policy + "\n")
    except PermissionError:  # the containers run as root
        subprocess.run(["sudo", "-n", "tee", path], input=(policy + "\n").encode(),
                       stdout=subprocess.DEVNULL, check=True)


def probe(chat, prompt, max_tokens):
    chunks = []
    r = chat(prompt, max_tokens, stream=True, on_chunk=lambda t, n: chunks.append((t, n)) and False)
    if len(chunks) < 3:
        return None
    (t0, n0), (t1, n1) = chunks[0], chunks[-1]
    rounds = len(chunks) - 1
    tokens = (n1 - n0) if n1 is not None and n0 is not None else None
    ms = 1000 * (t1 - t0) / rounds
    return {"ms": ms, "tok_round": tokens / rounds if tokens else None,
            "tps": tokens / (t1 - t0) if tokens else None, "tokens": r["completion_tokens"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--depths", nargs="+", type=int)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=512)
    a = ap.parse_args()
    port = config_value("PORT")
    os.environ.setdefault("GB10_BASE_URL", f"http://127.0.0.1:{port}/v1")
    from common import chat  # noqa: E402 (needs GB10_BASE_URL first)

    depth_file = os.path.join(config_value("CACHE_DIR"), "dspark_depth")
    depths = a.depths if a.depths is not None else list(range(int(config_value("DRAFT_TOKENS")) + 1))
    if not os.path.exists(depth_file):
        sys.exit(f"no {depth_file}: boot with DEPTH_PROBE=1 ./serve.sh first")

    def run(policy):
        write_policy(depth_file, policy)
        # >= 64 verify steps so rank 0 re-reads the file before the measured requests
        chat("Count from 1 to 300, one number a line.", 600, stream=True)
        time.sleep(1)
        rows = {}
        for name, prompt in PROMPTS.items():
            rs = [x for x in (probe(chat, prompt, a.max_tokens) for _ in range(a.reps)) if x]
            rows[name] = rs
            if rs:
                print(f"   {policy[:18]:18} {name:10} {statistics.median(x['tps'] or 0 for x in rs):7.1f} tok/s  "
                      f"{statistics.median(x['tok_round'] or 0 for x in rs):5.2f} tok/round  "
                      f"{statistics.median(x['ms'] for x in rs):6.2f} ms/round", flush=True)
        return rows

    cost = {}
    for d in depths:
        rows = run(f"fixed:{d}")
        ms = [statistics.median(x["ms"] for x in rs) for rs in rows.values() if rs]
        if ms:
            cost[d] = statistics.median(ms)
    if sorted(cost) != list(range(len(cost))):
        sys.exit(f"depths measured: {sorted(cost)}; the table needs 0..N without gaps")
    table = ",".join(f"{cost[d]:.1f}" for d in sorted(cost))
    run(f"conf:{table}")
    write_policy(depth_file, f"conf:{table}")
    print(f"\nROUND_MS={table}   (depths 0..{max(cost)}; the running server keeps it until restart)")
    print(f"next boots: ROUND_MS={table} ./serve.sh, or ROUND_MS=\"${{ROUND_MS:-{table}}}\" in config.local.sh")


if __name__ == "__main__":
    main()
