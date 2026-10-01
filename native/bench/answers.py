#!/usr/bin/env python3
"""Collect a model's answers to a set of questions, then compare runs side by side.

    # once per served model (restart the server between them):
    python3 bench/answers.py collect questions.jsonl --out runs/base.jsonl
    python3 bench/answers.py collect questions.jsonl --out runs/triz.jsonl
    # one Markdown file, every question with each run's answer under it:
    python3 bench/answers.py compare runs/base.jsonl runs/triz.jsonl > compare.md

For judging a fine-tune by reading, not by a score: same questions, same
settings (temperature 0 by default), one page. Endpoint, model and key come
from GB10_BASE_URL / GB10_MODEL / GB10_API_KEY as for the other benches.

questions: JSON lines (or one JSON array) in any of the common shapes; a
reference answer, when there is one, is kept and shown first:
  {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
      (a trailing assistant turn is the reference; the rest is sent as is)
  {"conversations": [{"from": "human", "value": ...}, {"from": "gpt", "value": ...}]}
  {"instruction": ..., "input": ..., "output": ...}   (Alpaca)
  {"question" | "prompt" | "query": ..., "answer" | "output" | "response": ...}
A line that is not JSON is a question on its own.

collect options: --thinking (off by default), --effort medium, --max-tokens,
--temperature, --limit N, --parallel N. Thinking, when on, is kept apart
(reasoning_content) and folded under the answer in the comparison.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROLE = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant",
        "system": "system", "model": "assistant", "bot": "assistant"}


def _messages(conv):
    msgs = []
    for m in conv:
        role = ROLE.get(str(m.get("role") or m.get("from") or "").lower())
        text = m.get("content", m.get("value"))
        if role is None or text is None:
            raise ValueError(f"unknown turn {m!r:.80}")
        if isinstance(text, list):  # OpenAI content parts
            text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
        msgs.append({"role": role, "content": text})
    return msgs


def to_item(rec, n):
    """One record -> {"id", "messages", "reference"}."""
    if isinstance(rec, str):
        return {"id": n, "messages": [{"role": "user", "content": rec}], "reference": None}
    rid = rec.get("id", n)
    conv = rec.get("messages") or rec.get("conversations")
    if conv:
        msgs = _messages(conv)
        ref = None
        if msgs and msgs[-1]["role"] == "assistant":
            ref = msgs.pop()["content"]
        if not msgs or msgs[-1]["role"] != "user":
            raise ValueError(f"record {rid}: does not end with a user turn")
        return {"id": rid, "messages": msgs, "reference": ref}
    q = rec.get("question") or rec.get("prompt") or rec.get("query")
    if q is None and rec.get("instruction") is not None:
        q = rec["instruction"] + (("\n\n" + rec["input"]) if rec.get("input") else "")
    if q is None:
        raise ValueError(f"record {rid}: no question field among {sorted(rec)}")
    msgs = [{"role": "system", "content": rec["system"]}] if rec.get("system") else []
    msgs.append({"role": "user", "content": q})
    ref = rec.get("answer") or rec.get("output") or rec.get("response") or rec.get("reference")
    return {"id": rid, "messages": msgs, "reference": ref}


def load_questions(path):
    text = Path(path).read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        records = json.loads(text)
    else:
        records = []
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except ValueError:
                records.append(line.strip())
    return [to_item(r, i) for i, r in enumerate(records)]


def ask(common, item, args):
    kwargs = {"enable_thinking": args.thinking}
    if args.effort:
        kwargs["reasoning_effort"] = args.effort
    body = {"model": common.MODEL, "messages": item["messages"], "max_tokens": args.max_tokens,
            "temperature": args.temperature, "chat_template_kwargs": kwargs}
    req = urllib.request.Request(common.CHAT_URL, json.dumps(body).encode(), common.HEADERS)
    t0 = time.perf_counter()
    try:
        d = json.loads(urllib.request.urlopen(req, timeout=3600).read())
    except urllib.error.HTTPError as e:
        return {**item, "error": f"HTTP {e.code}: {e.read()[:300]!r}"}
    msg = d["choices"][0]["message"]
    return {**item, "answer": msg.get("content") or "",
            "reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "",
            "finish_reason": d["choices"][0].get("finish_reason"),
            "completion_tokens": (d.get("usage") or {}).get("completion_tokens"),
            "seconds": round(time.perf_counter() - t0, 1)}


def collect(args):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import common  # reads the endpoint; only needed here

    items = load_questions(args.questions)[: args.limit or None]
    info = common.server_info()
    header = {"_run": {"model": common.MODEL, "model_path": info.get("model_path"),
                       "thinking": args.thinking, "effort": args.effort,
                       "temperature": args.temperature, "max_tokens": args.max_tokens,
                       "questions": str(args.questions), "started": time.strftime("%FT%TZ", time.gmtime())}}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"{len(items)} questions -> {out}  ({common.MODEL}, {info.get('model_path', '?')})")
    with out.open("w", encoding="utf-8") as f, ThreadPoolExecutor(args.parallel) as pool:
        f.write(json.dumps(header, ensure_ascii=False) + "\n")
        for i, r in enumerate(pool.map(lambda it: ask(common, it, args), items), 1):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            state = r.get("error") or f"{r['completion_tokens']} tok, {r['seconds']} s, {r['finish_reason']}"
            print(f"  [{i}/{len(items)}] {r['id']}: {state}", flush=True)
    return 0


def read_run(path):
    header, rows = {}, {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if "_run" in r:
            header = r["_run"]
        else:
            rows[str(r["id"])] = r
    return header, rows


def quote(text):
    return "\n".join("> " + ln for ln in (text or "").splitlines()) or ">"


def compare(args):
    runs = [(Path(p).stem, *read_run(p)) for p in args.runs]
    ids = list(runs[0][2])
    w = sys.stdout.write
    w("# Answers side by side\n\n")
    for name, h, _ in runs:
        w(f"- **{name}**: {h.get('model_path') or h.get('model')}, thinking "
          f"{'on' if h.get('thinking') else 'off'}{', effort ' + h['effort'] if h.get('effort') else ''}, "
          f"temperature {h.get('temperature')}\n")
    for qid in ids:
        first = runs[0][2][qid]
        w(f"\n---\n\n## {qid}\n\n")
        for m in first["messages"]:
            w(f"**{m['role']}:**\n\n{quote(m['content'])}\n\n")
        if first.get("reference"):
            w(f"<details><summary>reference answer</summary>\n\n{first['reference']}\n\n</details>\n\n")
        for name, _, rows in runs:
            r = rows.get(qid)
            if r is None:
                w(f"### {name}\n\n(missing)\n\n")
                continue
            if r.get("error"):
                w(f"### {name}\n\nerror: {r['error']}\n\n")
                continue
            w(f"### {name} ({r.get('completion_tokens')} tok, {r.get('finish_reason')})\n\n")
            if r.get("reasoning"):
                w(f"<details><summary>thinking</summary>\n\n{r['reasoning']}\n\n</details>\n\n")
            w(f"{r['answer']}\n\n")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("questions")
    c.add_argument("--out", required=True)
    c.add_argument("--thinking", action="store_true")
    c.add_argument("--effort", default=None)
    c.add_argument("--max-tokens", type=int, default=4096)
    c.add_argument("--temperature", type=float, default=0.0)
    c.add_argument("--limit", type=int, default=0)
    c.add_argument("--parallel", type=int, default=4)
    m = sub.add_parser("compare")
    m.add_argument("runs", nargs="+")
    args = ap.parse_args()
    return collect(args) if args.cmd == "collect" else compare(args)


if __name__ == "__main__":
    sys.exit(main())
