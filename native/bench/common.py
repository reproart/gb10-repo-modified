"""Shared client helpers for the GB10 benchmark scripts.

Endpoint, model and key come from the environment so nothing is baked in, and
one command shape measures any server:

    GB10_BASE_URL=http://127.0.0.1:8888/v1 python3 bench/perf.py

Env:
  GB10_BASE_URL  OpenAI-compatible API root INCLUDING /v1
                 (default http://127.0.0.1:8888/v1, ./serve.sh)
  GB10_MODEL     served model name; unset -> read from GB10_BASE_URL/models
                 ("default" if the server lists it, else the first entry)
  GB10_API_KEY   bearer token (default none; serve.sh's API_KEY if set)
  GB10_METRICS_URL  Prometheus endpoint of the ENGINE (default: GB10_BASE_URL
                 without /v1, plus /metrics). Behind a gateway, point it at
                 the engine itself. SGLang needs --enable-metrics (serve.sh
                 sets it). Unreachable -> the metric columns just show n/a.
                 Its host also serves /get_server_info (SGLang), printed as
                 the run header.
  GB10_GPU_WATCH  1 / 0: sample GPU temperature, SM clock and throttle reasons
                 with the local nvidia-smi (default: on when GB10_BASE_URL is
                 this host and nvidia-smi exists)
  GB10_RUN_DIR   where perf.py / longctx.py save a copy of their output
                 (default results/runs; empty = don't save)
  GB10_METRICS_PORT  optional engine-side metrics sidecar on the same host
                 (PLE counters of a patched vLLM; default 0 = off)
  GB10_METRICS_INTERVAL  the sidecar's refresh period (default 5 s); settled
                 reads wait one period + 1 s

Everything goes through /v1/chat/completions — the path real clients use, so
the chat template and the reasoning parser are part of the measurement.
Token counts always come from the server's `usage`, never from counting SSE
events: DFlash2 emits ~3.75 tokens per event (MTP several too), so
event-counting inflates the rate by roughly 4x.
"""
import datetime
import itertools
import json
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

BASE_URL = os.environ.get("GB10_BASE_URL", "http://127.0.0.1:8888/v1").rstrip("/")
API_KEY = os.environ.get("GB10_API_KEY", "")

CHAT_URL = f"{BASE_URL}/chat/completions"
SERVER_ROOT = BASE_URL[:-3] if BASE_URL.endswith("/v1") else BASE_URL
METRICS_URL = os.environ.get("GB10_METRICS_URL") or f"{SERVER_ROOT}/metrics"
ENGINE_ROOT = METRICS_URL[:-len("/metrics")] if METRICS_URL.endswith("/metrics") else SERVER_ROOT
HEADERS = {"Content-Type": "application/json"}
if API_KEY:
    HEADERS["Authorization"] = f"Bearer {API_KEY}"

# The canonical decode-throughput prompt used for every figure in the README.
CODE_PROMPT = (
    "Write a complete Python implementation of an LRUCache class with get and put "
    "in O(1), using a dict and a doubly linked list. Include docstrings."
)


def _get(url, timeout=30):
    req = urllib.request.Request(url, headers=HEADERS)
    return urllib.request.urlopen(req, timeout=timeout).read().decode()


def _model():
    if os.environ.get("GB10_MODEL"):
        return os.environ["GB10_MODEL"]
    try:
        ids = [m["id"] for m in json.loads(_get(f"{BASE_URL}/models"))["data"]]
    except Exception as e:  # noqa: BLE001 - report the actual cause
        raise SystemExit(f"cannot read {BASE_URL}/models ({e}); is the server up, "
                         f"is GB10_API_KEY right? Or set GB10_MODEL explicitly.") from None
    if not ids:
        raise SystemExit(f"{BASE_URL}/models lists no models; set GB10_MODEL explicitly.")
    return "default" if "default" in ids else ids[0]


MODEL = _model()


# ---- unique prompts ----------------------------------------------------------

def _make_vocab(n=30000, seed=1234):
    """A fixed synthetic vocabulary of n pronounceable pseudo-words."""
    rng = random.Random(seed)
    cons, vows = "bcdfghjklmnprstvwz", "aeiou"
    words = set()
    while len(words) < n:
        k = rng.choice((1, 2, 2, 3, 3, 4))
        words.add("".join(rng.choice(cons) + rng.choice(vows) for _ in range(k)))
    return sorted(words)


# Zipf-distributed word draws (weight 1/rank^1.07, like natural language): a
# realistic spread of distinct n-grams. A few dozen fixed words have so few
# distinct n-grams that anything keyed on them (n-gram tables, page cache)
# looks cheaper than it is on real text.
_VOCAB = _make_vocab()
_ZIPF_CUM = list(itertools.accumulate(1.0 / (r + 1) ** 1.07 for r in range(len(_VOCAB))))
_TOK_PER_WORD = [None]
_TOK_LOCK = threading.Lock()


def calibrate():
    """Calibrate words -> tokens once from the server's own `usage`: the prompt
    token difference between a 2000-word and a one-word request. Works on any
    OpenAI-compatible server (no /tokenize needed); falls back to 2.0.

    Call it before anything is measured: done lazily, its two requests (the
    2000-word one is the same text every run, so a prefix-cache hit on repeat
    runs) would land inside the first measured interval."""
    with _TOK_LOCK:
        if _TOK_PER_WORD[0] is None:
            sample = " ".join(random.Random(7).choices(_VOCAB, cum_weights=_ZIPF_CUM, k=2000))
            try:
                big = chat(f"{sample}\n\nReply with only: OK", 1)["prompt_tokens"]
                small = chat("x\n\nReply with only: OK", 1)["prompt_tokens"]
                _TOK_PER_WORD[0] = max(0.5, (big - small) / 2000)
            except Exception:  # noqa: BLE001 - calibration is best-effort
                _TOK_PER_WORD[0] = 2.0
    return _TOK_PER_WORD[0]


def unique_prompt(seed, approx_tokens, tail="Reply with only: OK"):
    """A prompt nothing else shares — not even its first block.

    The seed is the very first text, so the prefix/radix cache cannot reuse a
    single block across prompts; build prompts from shared filler instead and
    you measure cache hits rather than prefill or KV capacity. The body is
    Zipf-drawn words from a 30k-word vocabulary.
    """
    rng = random.Random(seed)
    words = int(approx_tokens / calibrate())
    body = " ".join(rng.choices(_VOCAB, cum_weights=_ZIPF_CUM, k=words))
    return f"Document {seed}. Below is a log excerpt.\n\n{body}\n\n{tail}"


def cache_hit_significant(hits, prompt_tokens):
    """True when a prefix-cache hit on a unique prompt is more than the shared
    chat-template head (a handful of tokens every prompt starts with)."""
    return bool(hits) and hits > max(64, 0.01 * (prompt_tokens or 0))


# ---- chat --------------------------------------------------------------------

def broken(text, expect=None):
    """Why an answer looks like garbage, '' if it does not: one character or one word
    making up most of it (a model whose logits went NaN repeats token 0, often '!'),
    or, with `expect`, that string missing (perf.py's prefill prompts ask for OK).
    Speed measured on such answers means nothing."""
    from collections import Counter

    t = (text or "").strip()
    if expect is not None:
        return "" if expect.lower() in t.lower() else f"answered {t[:40]!r}, not {expect!r}"
    chars = [c for c in t if not c.isspace()]
    if len(chars) >= 20:
        c, n = Counter(chars).most_common(1)[0]
        if n / len(chars) > 0.6:
            return f"{100 * n // len(chars)}% of it is {c!r} ({t[:30]!r}...)"
    words = t.split()
    if len(words) >= 12:
        w, n = Counter(words).most_common(1)[0]
        if n / len(words) > 0.7:
            return f"{100 * n // len(words)}% of its words are {w[:20]!r}"
    return ""


def chat(prompt, max_tokens, thinking=False, stream=False, timeout=3600, effort=None,
         ignore_eos=False, on_chunk=None):
    """One chat completion -> dict(e2e, ttft, completion_tokens, prompt_tokens,
    cached_tokens, finish_reason, content, stopped). ttft is None when not
    streaming; streamed, content holds the first ~8000 characters (answer and
    reasoning), enough for broken() to tell an answer from garbage.

    on_chunk(t, completion_tokens) (streaming only) is called on every chunk
    that carries text, with its perf_counter() time and the completion tokens
    so far (per-chunk usage, `continuous_usage_stats`, which SGLang and vLLM
    both send; None if the server does not). Returning True stops reading:
    the connection closes, the server aborts the request, and the result has
    stopped=True with the counts of the last chunk.

    `effort` maps to the template's reasoning_effort kwarg (xhigh is the
    template default; medium and low also exist, "high" does not). Ignored by
    the template when thinking is off.

    ignore_eos=True (vLLM / SGLang extension) forces exactly max_tokens tokens;
    a gateway may drop it — check completion_tokens.
    """
    template_kwargs = {"enable_thinking": thinking}
    if effort:
        template_kwargs["reasoning_effort"] = effort
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": template_kwargs,
        "stream": stream,
    }
    if stream:
        body["stream_options"] = {"include_usage": True}
        if on_chunk:
            body["stream_options"]["continuous_usage_stats"] = True
    if ignore_eos:
        body["ignore_eos"] = True
    req = urllib.request.Request(CHAT_URL, json.dumps(body).encode(), HEADERS)
    t0 = time.perf_counter()
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read()[:300]!r}") from None

    ttft, usage, finish, content, stopped = None, {}, None, "", False
    with resp:
        if not stream:
            d = json.loads(resp.read())
            usage = d.get("usage") or {}
            finish = d["choices"][0].get("finish_reason")
            content = d["choices"][0]["message"].get("content") or ""
        else:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    d = json.loads(payload)
                except ValueError:
                    continue
                if d.get("error"):
                    # SGLang / vLLM / LiteLLM report a mid-stream failure as an
                    # SSE event; ignoring it would count a dead request as a
                    # successful one with zero tokens.
                    raise RuntimeError(f"stream error: {json.dumps(d['error'])[:300]}")
                text = False
                for ch in d.get("choices") or []:
                    delta = ch.get("delta") or {}
                    piece = (delta.get("content") or delta.get("reasoning_content")
                             or delta.get("reasoning"))
                    if piece:
                        text = True
                        if len(content) < 8000:
                            content += piece
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                    finish = ch.get("finish_reason") or finish
                if d.get("usage"):
                    usage = d["usage"]
                if on_chunk and text and on_chunk(time.perf_counter(),
                                                  (d.get("usage") or {}).get("completion_tokens")):
                    stopped = True
                    break
    if not usage and not stopped:
        raise RuntimeError("response carried no `usage` — token counts would read as 0 "
                           "(does the endpoint pass stream_options.include_usage through?)")
    details = usage.get("prompt_tokens_details") or {}
    return {
        "e2e": time.perf_counter() - t0,
        "ttft": ttft,
        "completion_tokens": usage.get("completion_tokens", 0),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        # Only reported when the server is asked to (vLLM
        # --enable-prompt-tokens-details, SGLang --enable-cache-report);
        # otherwise None and the prefix-cache counters from metrics() are used.
        "cached_tokens": details.get("cached_tokens"),
        "finish_reason": finish,
        "content": content,
        "stopped": stopped,
    }


def decode_rate(r):
    """Generation speed after the first token (tok/s), None if unknown."""
    if r["ttft"] is None or r["e2e"] <= r["ttft"] or r["completion_tokens"] < 2:
        return None
    return (r["completion_tokens"] - 1) / (r["e2e"] - r["ttft"])


# ---- engine metrics ----------------------------------------------------------

# key -> candidate metric names (vLLM and SGLang); the first one present wins.
_COUNTERS = {
    "queue_sum": ("vllm:request_queue_time_seconds_sum", "sglang:queue_time_seconds_sum"),
    "queue_count": ("vllm:request_queue_time_seconds_count", "sglang:queue_time_seconds_count"),
    "pc_hits": ("vllm:prefix_cache_hits_total", "sglang:cached_tokens_total"),
    "drafts": ("vllm:spec_decode_num_drafts_total",),
    "draft_tokens": ("vllm:spec_decode_num_draft_tokens_total",),
    "accepted": ("vllm:spec_decode_num_accepted_tokens_total",),
    "preempted": ("vllm:num_preemptions_total",),
}
# Gauges (current value, not counters): max over label sets, not sum.
_GAUGES = {
    "kv_usage": ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc",
                 "sglang:token_usage"),
    "running": ("vllm:num_requests_running", "sglang:num_running_reqs"),
    "waiting": ("vllm:num_requests_waiting", "sglang:num_queue_reqs"),
    "accept_len": ("sglang:spec_accept_length",),
    "start_time": ("process_start_time_seconds",),
}


def _scrape(text, names, agg):
    for name in names:
        vals = re.findall(rf"^{re.escape(name)}(?:{{[^}}]*}})?\s+([0-9.eE+-]+)$", text, re.M)
        if vals:
            return agg(float(v) for v in vals)
    return None


def metrics():
    """Selected engine counters and gauges from METRICS_URL; {} if unreachable
    — the benches then just omit the derived columns."""
    try:
        text = _get(METRICS_URL, timeout=10)
    except Exception:  # noqa: BLE001 - metrics are optional
        return {}
    out = {}
    for table, agg in ((_COUNTERS, sum), (_GAUGES, max)):
        for key, names in table.items():
            v = _scrape(text, names, agg)
            if v is not None:
                out[key] = v
    return out


def delta(m0, m1, key):
    if key in m0 and key in m1:
        return m1[key] - m0[key]
    return None


def restarted(m0, m1):
    """True when the engine process changed between two metrics() reads — a
    supervisor restart mid-run makes everything after it a cold-server number."""
    return "start_time" in m0 and "start_time" in m1 and m1["start_time"] > m0["start_time"] + 1


def spec_summary(m0, m1):
    """Speculative-decoding summary for a metrics() pair, '' when unavailable.

    vLLM exposes counters (exact over the interval); SGLang only a gauge of the
    recent average accept length, read at the end of the interval."""
    d, a, dt = delta(m0, m1, "drafts"), delta(m0, m1, "accepted"), delta(m0, m1, "draft_tokens")
    if d and a is not None and dt:
        return f"spec {1 + a / d:.2f} tok/step, accept {100 * a / dt:.0f}%"
    if m1.get("accept_len"):
        return f"accept_len {m1['accept_len']:.2f}"
    return ""


class PeakWatch:
    """Samples KV-pool usage, running and waiting requests every `every` s in
    the background; `with PeakWatch() as p:` then read p.peak afterwards."""

    def __init__(self, every=1.0):
        self.every, self.peak = every, {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.wait(self.every):
            m = metrics()
            for key in ("kv_usage", "running", "waiting"):
                if m.get(key) is not None:
                    self.peak[key] = max(self.peak.get(key, 0.0), m[key])

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


# ---- run header ---------------------------------------------------------------

_SERVER_KEYS = (
    "version", "model_path", "revision", "speculative_algorithm",
    "speculative_draft_model_path", "speculative_draft_model_revision",
    "speculative_num_draft_tokens", "max_running_requests", "max_mamba_cache_size",
    "cuda_graph_max_bs", "max_total_tokens", "chunked_prefill_size", "context_length",
    "mem_fraction_static", "kv_cache_dtype", "mamba_ssm_dtype",
)


def server_info():
    """The engine's effective settings (SGLang /get_server_info), {} elsewhere.

    Recorded with every run because the README's worst traps are invisible in
    the numbers: which checkpoint revision actually loaded, and which draft
    token count (16 wins single-stream, 10 wins concurrency) was in effect."""
    try:
        d = json.loads(_get(f"{ENGINE_ROOT}/get_server_info", timeout=10))
    except Exception:  # noqa: BLE001 - optional
        return {}
    return {k: d[k] for k in _SERVER_KEYS if d.get(k) is not None}


def print_header(extra=""):
    print(f"endpoint {BASE_URL}  model {MODEL}{extra}")
    print(f"started {datetime.datetime.now(datetime.timezone.utc):%Y-%m-%dT%H:%M:%SZ}")
    info = server_info()
    if info:
        print("server " + "  ".join(f"{k}={v}" for k, v in info.items()))
    else:
        print(f"server settings unknown ({ENGINE_ROOT}/get_server_info unreachable)")
    if not metrics():
        print(f"note: {METRICS_URL} unreachable — queue / spec / prefix-cache "
              "columns show n/a (set GB10_METRICS_URL; SGLang needs --enable-metrics)")
    if gpu_watch_enabled():
        print("GPU: sampling the local nvidia-smi (GB10_GPU_WATCH=0 to disable)")
    mb = mem_blocks()
    if mb:
        print(f"memory: free {mb[0]:.1f} GiB, {mb[1]:.1f} GiB of it in 2 MiB+ blocks, up {mb[2]:.1f} days")
        note = last_mem_prep()
        if note:
            print(f"        {note}")


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for st in self.streams:
            st.write(data)

    def flush(self):
        for st in self.streams:
            st.flush()


def save_output(script):
    """Mirror stdout into GB10_RUN_DIR/<UTC time>-<script>-<model>.txt."""
    run_dir = os.environ.get("GB10_RUN_DIR",
                             os.path.join(os.path.dirname(__file__), "..", "results", "runs"))
    if not run_dir:
        return None
    os.makedirs(run_dir, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(run_dir, f"{stamp}-{script}-{re.sub(r'[^A-Za-z0-9._-]', '_', MODEL)}.txt")
    sys.stdout = _Tee(sys.stdout, open(path, "w", buffering=1))
    return os.path.abspath(path)


# ---- GPU temperature / clocks (local nvidia-smi) -------------------------------

# Throttle bits that mean the GPU was slowed: SW power cap, HW slowdown,
# SW / HW thermal slowdown, HW power brake. Idle (0x1) and app-clock (0x2)
# settings are normal between requests.
# Clock-event bits that mean "held below its clock": SW power cap, HW slowdown,
# SW / HW thermal slowdown, HW power brake.
_THROTTLE_MASK = 0x4 | 0x8 | 0x20 | 0x40 | 0x80
_REASON_NAMES = ((0x4, "SW power cap"), (0x8, "HW slowdown"), (0x20, "SW thermal"),
                 (0x40, "HW thermal"), (0x80, "HW power brake"))
# The GPU's own thermal limit, as nvidia-smi reports it: temperature.gpu.tlimit
# is the margin (°C) left to the T.Limit where it starts slowing itself down.
# On GB10 it follows a hotter sensor than temperature.gpu (40 °C of margin at
# 55 °C, 8 °C at 81 °C), so the margin is what to watch. Warn when it gets this
# small. (Earlier versions flagged 80 °C as a "suspend threshold": a Spark ran
# prefill at 84-87 °C with zero thermal-slowdown time on its counters.)
GPU_TLIMIT_WARN = 5
_GPU_QUERIES = ("temperature.gpu,clocks.sm,clocks_event_reasons.active,temperature.gpu.tlimit",
                "temperature.gpu,clocks.sm,clocks_event_reasons.active",
                "temperature.gpu,clocks.sm,clocks_throttle_reasons.active")
_GPU_QUERY = [None]
# Microseconds spent held down, by reason, since boot (newer drivers).
_GPU_COUNTERS = ("clocks_event_reasons_counters.sw_power_cap",
                 "clocks_event_reasons_counters.sw_thermal_slowdown",
                 "clocks_event_reasons_counters.hw_thermal_slowdown",
                 "clocks_event_reasons_counters.hw_power_brake_slowdown")
_GPU_COUNTERS_OK = [None]


def gpu_counters():
    """(power-cap, thermal) µs held down since boot, or None where the driver has no counters."""
    if _GPU_COUNTERS_OK[0] is False:
        return None
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={','.join(_GPU_COUNTERS)}",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        v = [int(float(x)) for x in out.stdout.splitlines()[0].split(",")] if not out.returncode else None
    except Exception:  # noqa: BLE001 - optional
        v = None
    _GPU_COUNTERS_OK[0] = v is not None
    return None if v is None else (v[0] + v[3], v[1] + v[2])


def gpu_watch_enabled():
    flag = os.environ.get("GB10_GPU_WATCH")
    if flag is not None:
        return flag == "1" and shutil.which("nvidia-smi") is not None
    local = urlparse(BASE_URL).hostname in ("127.0.0.1", "localhost", "::1")
    return local and shutil.which("nvidia-smi") is not None


def vmstat():
    """The kernel's memory-pressure counters on this machine (/proc/vmstat):
    direct compaction stalls, pages scanned and allocation stalls in direct
    reclaim, pages swapped out; and what was read from storage (pgpgin, KiB)
    and the page faults that had to wait for it (pgmajfault), the pages the
    background reclaim (kswapd) freed and the page-cache pages read again
    soon after reclaim had dropped them (workingset_refault_file): on Qwen3.8-
    Flash-Next the n-gram table is read from the checkpoint's files as the
    GPU touches its rows, so a prefill's speed follows how much of it came
    from the disk rather than the page cache. {} when the server is not local (the same
    rule and GB10_VMSTAT=0/1 override as GB10_GPU_WATCH) or on another OS.
    The stalls land on the allocating thread, i.e. a prefill chunk."""
    flag = os.environ.get("GB10_VMSTAT")
    local = urlparse(BASE_URL).hostname in ("127.0.0.1", "localhost", "::1")
    if (flag is not None and flag != "1") or (flag is None and not local):
        return {}
    try:
        with open("/proc/vmstat") as f:
            text = f.read()
    except OSError:
        return {}
    out = {"compact_stall": 0, "direct_scan": 0, "alloc_stall": 0, "swap_out": 0,
           "disk_in_kb": 0, "major_faults": 0, "kswapd_steal": 0, "refault": 0}
    for line in text.splitlines():
        key, _, val = line.partition(" ")
        if not val.strip().isdigit():
            continue
        if key == "compact_stall":
            out["compact_stall"] = int(val)
        elif key.startswith("pgscan_direct"):
            out["direct_scan"] += int(val)
        elif key.startswith("allocstall"):
            out["alloc_stall"] += int(val)
        elif key == "pswpout":
            out["swap_out"] = int(val)
        elif key == "pgpgin":
            out["disk_in_kb"] = int(val)
        elif key == "pgmajfault":
            out["major_faults"] = int(val)
        elif key.startswith("pgsteal_kswapd"):
            out["kswapd_steal"] += int(val)
        elif key in ("workingset_refault_file", "workingset_refault"):
            out["refault"] = int(val)
    return out


def mem_blocks():
    """(free GiB, GiB of it in free blocks of 2 MiB or more, uptime in days) on
    this machine, from /proc/buddyinfo; None when the server is not local (the
    rule of vmstat(), GB10_VMSTAT overrides) or on another OS. Over days of
    uptime free memory splits into small blocks, and on GB10 a boot's decode
    speed follows how much is left in large ones (scripts/mem-prep.sh)."""
    flag = os.environ.get("GB10_VMSTAT")
    local = urlparse(BASE_URL).hostname in ("127.0.0.1", "localhost", "::1")
    if (flag is not None and flag != "1") or (flag is None and not local):
        return None
    try:
        ps = os.sysconf("SC_PAGE_SIZE")
        with open("/proc/buddyinfo") as f:
            rows = [line.split()[4:] for line in f if " zone " in line]
        with open("/proc/uptime") as f:
            up = float(f.read().split()[0]) / 86400
    except (OSError, ValueError):
        return None
    k = 0
    while ps * 2 ** k < 2 * 2 ** 20:
        k += 1
    free = big = 0
    for row in rows:
        for order, n in enumerate(row):
            b = int(n) * ps * 2 ** order
            free += b
            big += b if order >= k else 0
    return free / 2 ** 30, big / 2 ** 30, up


def last_mem_prep():
    """The last line scripts/mem-prep.sh logged here at a server start, as one
    short note, '' if there is none."""
    try:
        with open(os.path.expanduser("~/.cache/gb10/mem-prep.log")) as f:
            line = f.readlines()[-1].split()
    except (OSError, IndexError):
        return ""
    kv = dict(x.split("=", 1) for x in line[3:] if "=" in x)
    how = ("after the cache drop and compaction" if line[2] == "dropped"
           else "not dropped or compacted (scripts/mem-prep.sh sudoers)")
    return (f"at the last server start ({line[0]}): free {kv.get('free')} GiB, "
            f"{kv.get('big')} GiB in 2 MiB+ blocks {how}")


_MEM_KEYS = ("MemTotal", "MemFree", "MemAvailable", "Buffers", "Cached", "SwapCached", "AnonPages",
             "Slab", "PageTables", "KernelStack", "SwapTotal", "SwapFree")


def _local_counters():
    flag = os.environ.get("GB10_VMSTAT")
    local = urlparse(BASE_URL).hostname in ("127.0.0.1", "localhost", "::1")
    return not ((flag is not None and flag != "1") or (flag is None and not local))


def meminfo():
    """Where this machine's memory is, in bytes: avail (MemAvailable), cache (page cache,
    Cached: on GB10 servers mostly the mmapped PLE table), anon (process heaps), swap
    (in use), driver (MemTotal minus everything the kernel itemizes: on GB10 the CUDA
    allocations of a model server, which no process's RSS shows). {} when not local
    (vmstat()'s rule) or on another OS."""
    if not _local_counters():
        return {}
    try:
        v = {}
        with open("/proc/meminfo") as f:
            for line in f:
                key, _, rest = line.partition(":")
                if key in _MEM_KEYS:
                    v[key] = int(rest.split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        return {}
    if len(v) < len(_MEM_KEYS):
        return {}
    itemized = sum(v[k] for k in ("MemFree", "Buffers", "Cached", "SwapCached", "AnonPages", "Slab",
                                  "PageTables", "KernelStack"))
    return {"avail": v["MemAvailable"], "free": v["MemFree"], "cache": v["Cached"],
            "anon": v["AnonPages"], "swap": v["SwapTotal"] - v["SwapFree"],
            "driver": max(0, v["MemTotal"] - itemized)}


def psi_memory():
    """(some, full) microseconds stalled on memory since boot (/proc/pressure/memory):
    some = at least one task waited, full = all non-idle tasks did. None if unavailable."""
    if not _local_counters():
        return None
    try:
        with open("/proc/pressure/memory") as f:
            tot = {line.split()[0]: int(line.rsplit("total=", 1)[1]) for line in f if "total=" in line}
        return tot.get("some", 0), tot.get("full", 0)
    except (OSError, ValueError, IndexError):
        return None


class MemWatch:
    """Samples meminfo() every `every` s while active: at the end str() says how low
    available memory went, what had grown or shrunk at that moment against the start
    (driver = CUDA allocations, page cache, anon), how much went to swap and how long
    tasks stalled on memory (PSI). '' when not local."""

    def __init__(self, every=0.2):
        self.every, self.start, self.low, self.end, self.psi = every, {}, {}, {}, None
        self.low_free = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _take(self):
        m = meminfo()
        if m and (not self.low or m["avail"] < self.low["avail"]):
            self.low = m
        if m and (self.low_free is None or m["free"] < self.low_free):
            self.low_free = m["free"]
        return m

    def _run(self):
        while not self._stop.wait(self.every):
            self._take()

    def __enter__(self):
        self.start = self._take()
        self._psi0 = psi_memory()
        if self.start:
            self._thread.start()
        return self

    def __exit__(self, *exc):
        if self.start:
            self._stop.set()
            self._thread.join()
            self.end = self._take()
            p1 = psi_memory()
            if self._psi0 and p1:
                self.psi = (p1[0] - self._psi0[0], p1[1] - self._psi0[1])

    def __str__(self):
        if not self.start or not self.low or not self.end:
            return ""
        g = 2.0 ** 30
        a, lo = self.start, self.low

        def d(k):
            return f"{(lo[k] - a[k]) / g:+.1f}"
        s = (f"mem: avail {a['avail'] / g:.1f} -> low {lo['avail'] / g:.1f} GiB (there: driver {d('driver')}, "
             f"page cache {d('cache')}, anon {d('anon')} GiB); free {a['free'] / g:.1f} -> low "
             f"{self.low_free / g:.1f} GiB; swap {(self.end['swap'] - a['swap']) / g:+.2f} GiB")
        if self.psi:
            s += f"; stalled on memory {self.psi[1] / 1e6:.1f} s all tasks, {self.psi[0] / 1e6:.1f} s some"
        if lo["avail"] < 1.5 * g:
            s += (f"  !! available fell to {lo['avail'] / g:.1f} GiB: earlyoom stops the server below 1 GiB "
                  "(scripts/host-tuning.sh; ~8 GB without it)")
        return s


_SERVER_MARKS = (b"sglang.launch_server", b"vllm.entrypoints", b"vllm serve", b"bin/vllm",
                 b"llama-server")
_CG_KEYS = ("file", "anon", "pgsteal", "pgsteal_direct", "pgsteal_kswapd", "workingset_refault_file")


def server_cgroup():
    """The cgroup v2 directory of the local model server: of the first process whose
    command line names sglang.launch_server, vllm or llama-server (GB10_SERVER_PID
    overrides). None when not local (vmstat()'s rule), not found or not cgroup v2."""
    if not _local_counters():
        return None
    pids = [os.environ["GB10_SERVER_PID"]] if os.environ.get("GB10_SERVER_PID") else None
    if pids is None:
        pids = []
        try:
            for d in os.listdir("/proc"):
                if not d.isdigit() or int(d) == os.getpid():
                    continue
                try:
                    with open(f"/proc/{d}/cmdline", "rb") as f:
                        cmd = f.read().replace(b"\0", b" ")
                except OSError:
                    continue
                if any(m in cmd for m in _SERVER_MARKS):
                    pids.append(d)
        except OSError:
            return None
    for pid in sorted(pids, key=int):
        try:
            with open(f"/proc/{pid}/cgroup") as f:
                path = next((ln[3:].strip() for ln in f if ln.startswith("0::")), None)
        except OSError:
            continue
        if path is not None and os.path.exists(f"/sys/fs/cgroup{path}/memory.stat"):
            return f"/sys/fs/cgroup{path}"
    return None


def cgroup_mem(path):
    """{max, high (bytes, None = no limit), current, and memory.stat's file, anon,
    pgsteal* (pages), workingset_refault_file} of a cgroup v2 directory, {} on error.
    Reclaim inside a cgroup at its own limit (memory.high / memory.max, e.g. the
    MemoryMax of scripts/install-service.sh's unit) is counted only here, not in
    /proc/vmstat: page cache dropped that way is invisible in vm_summary()."""
    if not path:
        return {}
    out = {}
    try:
        for k in ("max", "high", "current"):
            with open(f"{path}/memory.{k}") as f:
                v = f.read().strip()
            out[k] = None if v == "max" else int(v)
        with open(f"{path}/memory.stat") as f:
            for line in f:
                key, _, val = line.partition(" ")
                if key in _CG_KEYS:
                    out[key] = int(val)
    except (OSError, ValueError):
        return {}
    return out


def cgroup_summary(path, c0, c1):
    """The server cgroup's limits and what it reclaimed between c0 and c1, one line;
    '' without data."""
    if not c0 or not c1:
        return ""
    g, page = 2.0 ** 30, os.sysconf("SC_PAGE_SIZE")

    def lim(v):
        return "none" if v is None else f"{v / g:.1f} GiB"
    steal = (c1.get("pgsteal", 0) - c0.get("pgsteal", 0)) * page / g
    direct = (c1.get("pgsteal_direct", 0) - c0.get("pgsteal_direct", 0)) * page / g
    refault = c1.get("workingset_refault_file", 0) - c0.get("workingset_refault_file", 0)
    return (f"server cgroup {path.removeprefix('/sys/fs/cgroup')}: max {lim(c1['max'])}, high "
            f"{lim(c1['high'])}, using {c1['current'] / g:.1f} GiB (page cache {c1.get('file', 0) / g:.1f});"
            f" it reclaimed {steal:.2f} GiB ({direct:.2f} by its own tasks), {refault} refaults")


def vm_summary(v0, v1):
    """The vmstat() delta as one short note, '' without counters."""
    if not v0 or not v1:
        return ""
    d = {k: v1[k] - v0.get(k, 0) for k in v1}

    def n(x):
        return f"{x / 1e6:.1f}M" if x >= 1e6 else f"{x / 1e3:.0f}K" if x >= 1e4 else str(x)
    disk = d.get("disk_in_kb", 0) / 2**20
    return (f"vm: {n(d['compact_stall'])} compaction stalls, {n(d['alloc_stall'])} reclaim stalls"
            f" ({n(d['direct_scan'])} pages scanned), {n(d['swap_out'])} swapped out;"
            f" read from disk {disk:.2f} GiB, {n(d.get('major_faults', 0))} major faults;"
            f" kswapd freed {d.get('kswapd_steal', 0) * os.sysconf('SC_PAGE_SIZE') / 2**30:.2f} GiB,"
            f" {n(d.get('refault', 0))} pages read again after reclaim")


def gpu_sample():
    """(temp °C, SM MHz, throttle bits, T.Limit margin °C or None) of GPU 0, or None."""
    for q in ([_GPU_QUERY[0]] if _GPU_QUERY[0] else _GPU_QUERIES):
        try:
            out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10)
        except Exception:  # noqa: BLE001 - optional
            return None
        if out.returncode:
            continue
        _GPU_QUERY[0] = q

        def num(x, base=10):
            try:
                return int(float(x)) if base == 10 else int(x, base)
            except ValueError:
                return None
        f = [x.strip() for x in out.stdout.splitlines()[0].split(",")]
        return num(f[0]), num(f[1]), num(f[2], 16), (num(f[3]) if len(f) > 3 else None)
    return None


class GpuWatch:
    """Peak temperature, lowest SM clock, smallest margin to the GPU's thermal
    limit, any clock-event bits, and (newer drivers) the share of the time the
    GPU was held at its power cap or slowed for heat, while active; a no-op
    when disabled. str() gives a short summary for a table row."""

    def __init__(self, every=2.0):
        self.enabled = gpu_watch_enabled()
        self.every, self.max_temp, self.min_clock, self.reasons = every, None, None, 0
        self.min_margin, self._c0, self._t0, self.held = None, None, None, None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _take(self):
        s = gpu_sample()
        if not s:
            return
        t, c, r, m = s
        if t is not None:
            self.max_temp = t if self.max_temp is None else max(self.max_temp, t)
        if c is not None:
            self.min_clock = c if self.min_clock is None else min(self.min_clock, c)
        if m is not None:
            self.min_margin = m if self.min_margin is None else min(self.min_margin, m)
        self.reasons |= (r or 0) & _THROTTLE_MASK

    def _run(self):
        while not self._stop.wait(self.every):
            self._take()

    def __enter__(self):
        if self.enabled:
            self._c0, self._t0 = gpu_counters(), time.perf_counter()
            self._thread.start()
        return self

    def __exit__(self, *exc):
        if self.enabled:
            self._stop.set()
            self._thread.join()
            self._take()
            c1, wall = gpu_counters(), time.perf_counter() - self._t0
            if self._c0 and c1 and wall > 0:
                self.held = tuple(min(1.0, (b - a) / 1e6 / wall) for a, b in zip(self._c0, c1))

    def __str__(self):
        if self.max_temp is None:
            return ""
        s = f"GPU {self.max_temp}°C / {self.min_clock} MHz"
        if self.min_margin is not None:
            s += f" ({self.min_margin}°C below its T.Limit)"
        if self.held and (self.held[0] >= 0.05 or self.held[1] > 0):
            s += f", power-capped {self.held[0]:.0%} of the time"
            if self.held[1] > 0:
                s += f", thermal slowdown {self.held[1]:.0%}"
        # The power cap alone is how GB10 runs any heavy load (its clocks are
        # set by the SoC's power budget), so only the rest is flagged.
        if self.reasons & ~0x4:
            s += "  !! " + ", ".join(n for b, n in _REASON_NAMES if self.reasons & b)
        elif self.reasons and not self.held:
            s += " [power cap]"
        if self.min_margin is not None and self.min_margin <= GPU_TLIMIT_WARN:
            s += f"  !! within {self.min_margin}°C of the GPU's thermal limit"
        return s


# ---- optional PLE sidecar (patched vLLM only) --------------------------------

_SIDECAR_PORT = int(os.environ.get("GB10_METRICS_PORT", "0") or 0)
_SIDECAR = (f"http://{urlparse(BASE_URL).hostname}:{_SIDECAR_PORT}/metrics"
            if _SIDECAR_PORT else None)
_PLE = {"ple_ops": "vllm:ple_mmap_ops_total", "ple_op_ms": "vllm:ple_mmap_op_ms_total",
        "ple_gather_ms": "vllm:ple_mmap_gather_ms_total"}
_TORCH = {"torch_reserved": "vllm:torch_cuda_reserved_bytes",
          "torch_peak": "vllm:torch_cuda_max_reserved_bytes",
          "torch_allocated": "vllm:torch_cuda_allocated_bytes"}
_SIDECAR_LAG = float(os.environ.get("GB10_METRICS_INTERVAL", "5") or 5) + 1.0
_SIDECAR_SEEN = [False]


def sidecar(settle=False):
    """PLE gather counters from the engine-side sidecar; {} when not configured
    or unreachable.

    The sidecar copies the engine's counters into Prometheus only every few
    seconds, so a read right after a short request can miss its ops (they
    then land in the NEXT measurement). settle=True waits one refresh period
    first — use it for every read that closes (or opens, after other
    traffic) a measured interval."""
    if not _SIDECAR:
        return {}
    if settle and _SIDECAR_SEEN[0]:
        time.sleep(_SIDECAR_LAG)
    try:
        text = urllib.request.urlopen(_SIDECAR, timeout=5).read().decode()
    except Exception:  # noqa: BLE001 - optional
        return {}
    out = {}
    for key, name in (*_PLE.items(), *_TORCH.items()):
        m = re.search(rf"^{re.escape(name)}\s+([0-9.eE+-]+)$", text, re.M)
        if m:
            out[key] = float(m.group(1))
    _SIDECAR_SEEN[0] = _SIDECAR_SEEN[0] or any(k in out for k in _PLE)
    return out


def torch_summary(s0, s1):
    """'torch reserved +3.6 GiB (peak +3.9)' over a sidecar() pair: what the engine's
    caching allocator took from the machine (vllm-flash-ar's sidecar), '' without it."""
    if "torch_reserved" not in s0 or "torch_reserved" not in s1:
        return ""
    g = 2.0 ** 30
    s = f"torch reserved {(s1['torch_reserved'] - s0['torch_reserved']) / g:+.1f} GiB"
    if "torch_peak" in s0 and "torch_peak" in s1:
        s += f" (peak {(s1['torch_peak'] - s0['torch_peak']) / g:+.1f}, now {s1['torch_reserved'] / g:.1f} GiB)"
    return s


def ple_summary(s0, s1):
    """'PLE 3.1 ms/op (gather 1.2)' over a sidecar() pair — per-workload cost.
    op = hash + gather + H2D, including the wait for the preceding layer."""
    ops = delta(s0, s1, "ple_ops")
    if not ops:
        return ""
    return (f"PLE {delta(s0, s1, 'ple_op_ms') / ops:.1f} ms/op "
            f"(gather {delta(s0, s1, 'ple_gather_ms') / ops:.1f})")
