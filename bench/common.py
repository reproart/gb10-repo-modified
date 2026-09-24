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

def chat(prompt, max_tokens, thinking=False, stream=False, timeout=3600, effort=None,
         ignore_eos=False):
    """One chat completion -> dict(e2e, ttft, completion_tokens, prompt_tokens,
    cached_tokens, finish_reason, content). ttft is None when not streaming;
    content is only collected when not streaming.

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
    if ignore_eos:
        body["ignore_eos"] = True
    req = urllib.request.Request(CHAT_URL, json.dumps(body).encode(), HEADERS)
    t0 = time.perf_counter()
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read()[:300]!r}") from None

    ttft, usage, finish, content = None, {}, None, ""
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
                for ch in d.get("choices") or []:
                    delta = ch.get("delta") or {}
                    if ttft is None and (delta.get("content") or delta.get("reasoning_content")
                                         or delta.get("reasoning")):
                        ttft = time.perf_counter() - t0
                    finish = ch.get("finish_reason") or finish
                if d.get("usage"):
                    usage = d["usage"]
    if not usage:
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
_THROTTLE_MASK = 0x4 | 0x8 | 0x40 | 0x80 | 0x100
# RESULTS.md: the box suspends at 80 °C; sustained prefill measured 74 °C.
GPU_TEMP_WARN = 78
_GPU_QUERIES = ("temperature.gpu,clocks.sm,clocks_event_reasons.active",
                "temperature.gpu,clocks.sm,clocks_throttle_reasons.active")
_GPU_QUERY = [None]


def gpu_watch_enabled():
    flag = os.environ.get("GB10_GPU_WATCH")
    if flag is not None:
        return flag == "1" and shutil.which("nvidia-smi") is not None
    local = urlparse(BASE_URL).hostname in ("127.0.0.1", "localhost", "::1")
    return local and shutil.which("nvidia-smi") is not None


def gpu_sample():
    """(temp °C, SM MHz, throttle bits) of GPU 0, or None."""
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
        return num(f[0]), num(f[1]), num(f[2], 16)
    return None


class GpuWatch:
    """Peak temperature, lowest SM clock and any throttle bits while active;
    a no-op when disabled. str() gives a short summary for a table row."""

    def __init__(self, every=2.0):
        self.enabled = gpu_watch_enabled()
        self.every, self.max_temp, self.min_clock, self.reasons = every, None, None, 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _take(self):
        s = gpu_sample()
        if not s:
            return
        t, c, r = s
        if t is not None:
            self.max_temp = t if self.max_temp is None else max(self.max_temp, t)
        if c is not None:
            self.min_clock = c if self.min_clock is None else min(self.min_clock, c)
        self.reasons |= (r or 0) & _THROTTLE_MASK

    def _run(self):
        while not self._stop.wait(self.every):
            self._take()

    def __enter__(self):
        if self.enabled:
            self._thread.start()
        return self

    def __exit__(self, *exc):
        if self.enabled:
            self._stop.set()
            self._thread.join()
            self._take()

    def __str__(self):
        if self.max_temp is None:
            return ""
        s = f"GPU {self.max_temp}°C / {self.min_clock} MHz"
        if self.reasons:
            s += f"  !! throttled (0x{self.reasons:x})"
        if self.max_temp >= GPU_TEMP_WARN:
            s += f"  !! within {80 - self.max_temp}°C of the 80°C suspend threshold"
        return s


# ---- optional PLE sidecar (patched vLLM only) --------------------------------

_SIDECAR_PORT = int(os.environ.get("GB10_METRICS_PORT", "0") or 0)
_SIDECAR = (f"http://{urlparse(BASE_URL).hostname}:{_SIDECAR_PORT}/metrics"
            if _SIDECAR_PORT else None)
_PLE = {"ple_ops": "vllm:ple_mmap_ops_total", "ple_op_ms": "vllm:ple_mmap_op_ms_total",
        "ple_gather_ms": "vllm:ple_mmap_gather_ms_total"}
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
    for key, name in _PLE.items():
        m = re.search(rf"^{re.escape(name)}\s+([0-9.eE+-]+)$", text, re.M)
        if m:
            out[key] = float(m.group(1))
    _SIDECAR_SEEN[0] = _SIDECAR_SEEN[0] or bool(out)
    return out


def ple_summary(s0, s1):
    """'PLE 3.1 ms/op (gather 1.2)' over a sidecar() pair — per-workload cost.
    op = hash + gather + H2D, including the wait for the preceding layer."""
    ops = delta(s0, s1, "ple_ops")
    if not ops:
        return ""
    return (f"PLE {delta(s0, s1, 'ple_op_ms') / ops:.1f} ms/op "
            f"(gather {delta(s0, s1, 'ple_gather_ms') / ops:.1f})")
