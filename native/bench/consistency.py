#!/usr/bin/env python3
"""Answer checks the speed numbers cannot show: tool-call examples, greedy determinism, prefix-cache hits.

    python3 bench/consistency.py                       all of it
    python3 bench/consistency.py --only fence          one of: fence, repeat
    python3 bench/consistency.py --lengths 4000 32000 --tokens 200

Works against any server here (SGLang, vLLM, TensorFold; GB10_BASE_URL as for perf.py).
After blazux/qwen3.8-Flash-DGX, which found each of these broken for Qwen3.8 on vLLM.

1. fence: with tools offered and thinking off, the model is asked to repeat a short text holding the
   tool-call syntax inside a ```xml block, then a line with DONE; and a line that names <function=NAME>
   mid-sentence. Both must come back as text, DONE included, with no tool call (non-streamed and
   streamed). Stock SGLang's qwen3_coder parser takes the example for a call and drops the rest
   (patches/gb10_tool_fences.py keeps it text). A control asks for the weather in Paris: that one
   must come back as a get_weather call, both ways.
2. repeat: per prompt length, one unique prompt sent greedily, one at a time: cold, twice more (prefix-
   cache hits), and cold again: after a cache flush where the server allows one (SGLang /flush_cache,
   vLLM /reset_prefix_cache in dev mode), else with a cache_salt of its own (vLLM: the same tokens,
   no reuse). Compared: the reply text and the first token's top-5 logprobs. Cached tokens come from
   usage.prompt_tokens_details (vLLM: --enable-prompt-tokens-details). Hybrid models cache their
   recurrent state only at block boundaries (vLLM on Flash-Next: every 1600 tokens, a hit reaching
   round_down(P, 1600) - 1600), so a ~2K prompt gets no hit there: the default lengths are 4K and 32K.
     cold vs cold  the same request twice from scratch: a difference is non-deterministic kernels
                   (on vLLM a GB10 top-k in the sparse attention dropped candidates)
     hit vs hit    the same cached state twice: should match like cold vs cold
     cold vs hit   a cache hit restores the state a cold prefill computed; a large first-token gap or
                   another first token is a wrong restore (on vLLM: an all-zero Mamba state)
   A small first-token gap with a later fork is numerics: a hit computes fewer tokens, in other chunks.
Exit code: the number of FAIL lines.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid

import common
from common import BASE_URL, CHAT_URL, ENGINE_ROOT, HEADERS, MODEL, SERVER_ROOT, unique_prompt

TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather", "description": "Current weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "write_file", "description": "Write a text file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                       "required": ["path", "content"]}}},
]
EXAMPLE = ("Here is how a call looks:\n```xml\n<tool_call>\n<function=get_weather>\n<parameter=city>\nBerlin\n"
           "</parameter>\n</function>\n</tool_call>\n```")
MIDLINE = "In this format the header <function=NAME> opens a call."
REPEAT = "Repeat the text between the two lines of ===== exactly, character for character, then write DONE " \
         "on a line of its own. Do not call any tool.\n=====\n{}\n====="
# Gaps in the first token's top-5 logprobs (nats): below SMALL is numerics, above LARGE a different state.
SMALL, LARGE = 0.05, 0.5
fails = 0


def post(body, stream, timeout=900):
    """One chat completion -> dict(content, tool_calls [(name, args)], finish, cached, top5)."""
    body = dict(body, model=MODEL, stream=stream)
    if stream:
        body["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(CHAT_URL, json.dumps(body).encode(), HEADERS)
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read()[:300]!r}") from None
    content, calls, finish, usage, top5 = [], {}, None, {}, None
    with resp:
        if not stream:
            d = json.loads(resp.read())
            ch = d["choices"][0]
            msg = ch.get("message") or {}
            content.append(msg.get("content") or "")
            for i, tc in enumerate(msg.get("tool_calls") or []):
                fn = tc.get("function") or {}
                calls[i] = [fn.get("name"), fn.get("arguments") or ""]
            finish, usage = ch.get("finish_reason"), d.get("usage") or {}
            lp = ((ch.get("logprobs") or {}).get("content") or [None])[0]
            top5 = lp and {t["token"]: t["logprob"] for t in lp.get("top_logprobs") or []}
        else:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: ") or line[6:] == "[DONE]":
                    continue
                d = json.loads(line[6:])
                if d.get("error"):
                    raise RuntimeError(f"stream error: {json.dumps(d['error'])[:300]}")
                usage = d.get("usage") or usage
                for ch in d.get("choices") or []:
                    delta = ch.get("delta") or {}
                    content.append(delta.get("content") or "")
                    for tc in delta.get("tool_calls") or []:
                        c = calls.setdefault(tc.get("index", 0), [None, ""])
                        fn = tc.get("function") or {}
                        c[0] = fn.get("name") or c[0]
                        c[1] += fn.get("arguments") or ""
                    finish = ch.get("finish_reason") or finish
    cached = ((usage.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0
    return {"content": "".join(content), "tool_calls": [tuple(calls[k]) for k in sorted(calls)],
            "finish": finish, "cached": cached, "top5": top5}


def verdict(ok, label, detail=""):
    global fails
    word = {True: "PASS", False: "FAIL", None: "WARN"}[ok]
    fails += ok is False
    print(f"   {word:4}  {label}" + (f"  {detail}" if detail else ""), flush=True)


def fence_section(tokens):
    print("## 1. Tool-call syntax as text (tools offered, thinking off)")
    base = {"tools": TOOLS, "temperature": 0, "max_tokens": tokens,
            "chat_template_kwargs": {"enable_thinking": False}}
    for name, text, must in (("fenced example", EXAMPLE, "```xml"), ("<function=NAME> mid-line", MIDLINE,
                                                                     "<function=NAME>")):
        for stream in (False, True):
            r = post(dict(base, messages=[{"role": "user", "content": REPEAT.format(text)}]), stream)
            how = "streamed" if stream else "not streamed"
            got = r["content"].strip().replace("\n", "\\n")
            if not r["tool_calls"] and "DONE" in r["content"] and must in r["content"]:
                verdict(True, f"{name}, {how}: text with DONE, no tool call")
            elif r["tool_calls"] and "DONE" not in r["content"]:
                verdict(False, f"{name}, {how}: came back as a tool call {r['tool_calls'][0][0]!r}, the rest "
                        "dropped (the parser; SGLang: patches/gb10_tool_fences.py, TOOL_FENCES=1)",
                        f"content {got[:80]!r}")
            else:
                verdict(None, f"{name}, {how}: the model did not repeat it as asked (not a parser verdict)",
                        f"content {got[:100]!r}, calls {r['tool_calls']}")
    for stream in (False, True):
        r = post(dict(base, messages=[{"role": "user", "content":
                                       "What is the weather in Paris right now? Use the tool."}]), stream)
        how = "streamed" if stream else "not streamed"
        call = next((c for c in r["tool_calls"] if c[0] == "get_weather"), None)
        if call and "paris" in call[1].lower():
            try:
                json.loads(call[1])
                verdict(True, f"control, {how}: get_weather {call[1]}")
            except ValueError:
                verdict(False, f"control, {how}: get_weather with arguments that are not JSON", repr(call[1]))
        elif not r["tool_calls"]:
            verdict(None, f"control, {how}: no tool call, the model answered in text",
                    repr(r["content"][:100]))
        else:
            verdict(False, f"control, {how}: unexpected tool calls", repr(r["tool_calls"]))
    print()


def flush_cache():
    """Empty the server's prefix cache; True when one of the known endpoints accepted it."""
    for url in (f"{ENGINE_ROOT}/flush_cache", f"{SERVER_ROOT}/reset_prefix_cache"):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, b"", HEADERS, method="POST"), timeout=60) as r:
                if r.status == 200:
                    time.sleep(1)
                    return True
        except Exception:  # noqa: BLE001 - not this server's endpoint
            continue
    return False


def compare(a, b):
    """(texts equal, first diverging char or None, first-token top-1 equal, max top-5 logprob gap or None)."""
    ta, tb = a["content"], b["content"]
    same = ta == tb
    fork = None if same else next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
    gap, top1 = None, None
    if a["top5"] and b["top5"]:
        top1 = max(a["top5"], key=a["top5"].get) == max(b["top5"], key=b["top5"].get)
        common_tokens = set(a["top5"]) & set(b["top5"])
        gap = max((abs(a["top5"][t] - b["top5"][t]) for t in common_tokens), default=float("inf"))
    return same, fork, top1, gap


def report(label, a, b, kind):
    same, fork, top1, gap = compare(a, b)
    g = "" if gap is None else f", first-token top-5 gap {gap:.4f}"
    if same:
        verdict(True, f"{label}: same text ({len(a['content'])} chars){g}")
        return
    where = f"differs from char {fork} of {max(len(a['content']), len(b['content']))}"
    if kind == "hit" and (top1 is False or (gap is not None and gap > LARGE)):
        verdict(False, f"{label}: {where}, and the first token already differs{g}: the cache hit "
                "does not restore what the cold prefill computed")
    elif top1 is False or (gap is not None and gap > SMALL and kind == "cold"):
        verdict(False, f"{label}: {where}{g}: not deterministic from the first token on")
    elif gap is None:
        verdict(None, f"{label}: {where}")
    elif kind == "cold":
        verdict(None, f"{label}: {where}{g}; the first token agrees: a late fork, not bit-exact run to run")
    else:
        verdict(None, f"{label}: {where}{g}; the first token agrees: numerics (a hit computes fewer tokens, "
                "in other chunks), a late fork")


def cold_again(body):
    """The same request with nothing reused: (result, how), or (None, why not)."""
    if flush_cache():
        return post(body, False), "after a cache flush"
    try:
        r = post(dict(body, cache_salt=uuid.uuid4().hex), False)
    except RuntimeError as e:
        return None, f"no cache flush endpoint, cache_salt refused ({str(e)[:60]})"
    if r["cached"]:
        return None, f"no cache flush endpoint, and cache_salt was ignored ({r['cached']} tokens reused)"
    return r, "with its own cache_salt"


def repeat_section(lengths, tokens):
    print("## 2. The same greedy request again: cold, repeats (cache hits), cold again (one at a time)")
    for n, approx in enumerate(lengths):
        prompt = unique_prompt(int(time.time()) * 10 + n, approx,
                               tail="Write a short poem that uses five words from the text above.")
        body = {"messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": tokens,
                "logprobs": True, "top_logprobs": 5, "chat_template_kwargs": {"enable_thinking": False}}
        try:
            runs = [post(body, False)]
        except RuntimeError as e:
            if "HTTP 4" not in str(e):
                raise
            print(f"   (logprobs refused: {str(e)[:80]}; texts only)")
            body = {k: v for k, v in body.items() if k not in ("logprobs", "top_logprobs")}
            runs = [post(body, False)]
        runs += [post(body, False) for _ in range(2)]
        cold, how = cold_again(body)
        cached = "/".join(str(r["cached"]) for r in runs + ([cold] if cold else []))
        print(f"   ~{approx} tokens: cached tokens per run {cached}; cold again {how}", flush=True)
        if runs[0]["top5"] is None:
            print("         (no logprobs from this server: texts only)")
        hit = bool(runs[1]["cached"] and runs[2]["cached"])
        if not hit:
            print("         (no cached tokens on the repeats: nothing reusable at this length, prefix caching "
                  "off, or not reported; the repeats are compared as cold runs)")
        if cold:
            report("cold vs cold      ", runs[0], cold, "cold")
        report("hit vs hit        " if hit else "repeat vs repeat  ", runs[1], runs[2], "cold")
        if hit:
            report("cold vs hit       ", runs[0], runs[1], "hit")
        else:
            report("first vs repeat   ", runs[0], runs[1], "cold")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", choices=["fence", "repeat"])
    ap.add_argument("--lengths", nargs="+", type=int, default=[4000, 32000],
                    help="approx. prompt sizes for the repeat section (default: 4000 32000)")
    ap.add_argument("--tokens", type=int, default=200, help="reply length (default 200)")
    args = ap.parse_args()
    saved = common.save_output("consistency")
    print(f"endpoint {BASE_URL}  model {MODEL}")
    print(f"started {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
    try:
        if args.only in (None, "fence"):
            fence_section(args.tokens)
        if args.only in (None, "repeat"):
            repeat_section(args.lengths, args.tokens)
    except RuntimeError as e:
        sys.exit(f"request failed: {e}")
    print(f"{fails} FAIL line(s)" if fails else "no FAIL lines")
    if saved:
        print(f"saved -> {saved}")
    sys.exit(min(fails, 100))


if __name__ == "__main__":
    main()
