"""CPU checks for gb10_tool_fences against SGLang's own qwen3_coder detector.

The detector's source file is loaded with stand-ins for the few SGLang
modules it imports (no torch, no GPU), from GB10_SGLANG_SRC (the sglang
package directory) or the newest ~/spark/venv-sglang-*. Each case runs not
streamed and streamed in several chunkings, through the stock detector and
the patched one: the bug cases must fail stock and pass patched, the real
calls must come out the same from both.

    python3 patches/test_gb10_tool_fences.py
"""
from __future__ import annotations

import ast
import dataclasses
import glob
import importlib.util
import json
import os
import sys
import types
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gb10_tool_fences as tf  # noqa: E402

TOOLS = [types.SimpleNamespace(type="function", function=types.SimpleNamespace(
    name=n, parameters={"type": "object", "properties": {p: {"type": "string"}}}))
    for n, p in (("get_weather", "city"), ("write_file", "content"))]


def sglang_src():
    if os.environ.get("GB10_SGLANG_SRC"):
        return os.environ["GB10_SGLANG_SRC"]
    found = sorted(glob.glob(os.path.expanduser("~/spark/venv-sglang-*/lib/python3*/site-packages/sglang")))
    return found[-1] if found else None


def load_detector(src):
    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        sys.modules[name] = m
        return m
    for p in ("sglang", "sglang.srt", "sglang.srt.entrypoints", "sglang.srt.entrypoints.openai",
              "sglang.srt.function_call"):
        mod(p).__path__ = []

    @dataclasses.dataclass
    class ToolCallItem:
        tool_index: int
        name: Optional[str] = None
        parameters: str = ""

    @dataclasses.dataclass
    class StreamingParseResult:
        normal_text: str = ""
        calls: list = dataclasses.field(default_factory=list)

    class BaseFormatDetector:
        def __init__(self):
            self._buffer, self.prev_tool_call_arr, self.current_tool_id = "", [], -1
            self.current_tool_name_sent, self.streamed_args_for_tool = False, []

        def finish(self, tools):
            return StreamingParseResult()

    mod("sglang.srt.function_call.core_types", ToolCallItem=ToolCallItem,
        StreamingParseResult=StreamingParseResult, _GetInfoFunc=object)
    mod("sglang.srt.function_call.base_format_detector", BaseFormatDetector=BaseFormatDetector)
    mod("sglang.srt.entrypoints.openai.protocol", Tool=object)
    mod("sglang.srt.function_call.utils", get_schema_properties=lambda p: p.get("properties", {}),
        infer_type_from_json_schema=lambda s: s.get("type") if isinstance(s, dict) else None,
        safe_literal_eval=ast.literal_eval)
    path = os.path.join(src, "srt", "function_call", "qwen3_coder_detector.py")
    spec = importlib.util.spec_from_file_location(tf.TARGET_MODULE, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[tf.TARGET_MODULE] = m
    spec.loader.exec_module(m)
    return m


def run(cls, text, chunk):
    """(normal text, [(name, args dict)]) from one detector; chunk 0 = not streamed."""
    d = cls()
    if chunk == 0:
        r = d.detect_and_parse(text, TOOLS)
        return r.normal_text, [(c.name, json.loads(c.parameters)) for c in r.calls]
    normal, calls = [], {}
    pieces = [text[i:i + chunk] for i in range(0, len(text), chunk)]
    results = [d.parse_streaming_increment(p, TOOLS) for p in pieces] + [d.finish(TOOLS)]
    for r in results:
        normal.append(r.normal_text or "")
        for c in r.calls or ():
            name, args = calls.get(c.tool_index, (None, ""))
            calls[c.tool_index] = (c.name or name, args + (c.parameters or ""))
    return "".join(normal), [(n, json.loads(a) if a else {}) for _, (n, a) in sorted(calls.items())]


CALL = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"
EXAMPLE = ("```xml\n<tool_call>\n<function=get_weather>\n<parameter=city>\nBerlin\n</parameter>\n</function>\n"
           "</tool_call>\n```\n")
CASES = [  # name, text, expected (normal, calls) with the patch, "bug" if stock gets it wrong
    ("fenced example, prose after", "Here is the format:\n" + EXAMPLE + "That is all.",
     ("Here is the format:\n" + EXAMPLE + "That is all.", []), "bug"),
    ("tilde fence", "See:\n~~~\n<tool_call>\n<function=get_weather>\n</function>\n</tool_call>\n~~~\nok",
     ("See:\n~~~\n<tool_call>\n<function=get_weather>\n</function>\n</tool_call>\n~~~\nok", []), "bug"),
    ("nested fence: ``` inside ````", "````md\n```xml\n<tool_call>\n```\n<tool_call>\n````\nok",
     ("````md\n```xml\n<tool_call>\n```\n<tool_call>\n````\nok", []), "bug"),
    ("<function= mid-line in prose", "The header <function=NAME> starts a call.\nDone.",
     ("The header <function=NAME> starts a call.\nDone.", []), "bug-stream"),
    ("real call after a closed fence", "Example:\n" + EXAMPLE + "Now:\n" + CALL,
     ("Example:\n" + EXAMPLE + "Now:\n", [("get_weather", {"city": "Paris"})]), "bug"),
    ("plain real call", CALL, (None, [("get_weather", {"city": "Paris"})]), "same"),
    ("text, then a real call", "Checking.\n" + CALL, ("Checking.\n", [("get_weather", {"city": "Paris"})]), "same"),
    ("a call whose value holds a fence",
     "<tool_call>\n<function=write_file>\n<parameter=content>\n# T\n```python\nprint(1)\n```\n</parameter>\n"
     "</function>\n</tool_call>", (None, [("write_file", {"content": "# T\n```python\nprint(1)\n```"})]), "same"),
    ("4-space indent is no fence", "x\n    ```\n" + CALL,
     ("x\n    ```\n", [("get_weather", {"city": "Paris"})]), "same"),
    ("unclosed fence keeps the guard on (known limit)", "```\n" + CALL, ("```\n" + CALL, []), "bug"),
    ("a lone < at the very end", "a < b <", ("a < b <", []), "stream-end"),
]
CHUNKS = (0, 1, 3, 7, 4096)


def main():
    fails = 0
    g = tf.Guard()
    assert g.feed("```xml\n") == "```xml\n" and g.fence == ("`", 3)
    assert g.feed("<tool_c") == "" and g.held == "<tool_c"
    assert g.feed("all>") == tf.HIDDEN + "tool_call>"
    assert g.feed("\n```\n") == "\n```\n" and g.fence is None
    print("PASS Guard: fence open / held tail / hidden marker / fence close")

    src = sglang_src()
    if not src or not os.path.exists(os.path.join(src, "srt", "function_call", "qwen3_coder_detector.py")):
        print("SKIP the detector cases: no SGLang source (GB10_SGLANG_SRC=<.../site-packages/sglang>)")
        return 0
    stock_mod = load_detector(src)
    stock = stock_mod.Qwen3CoderDetector
    patched = type("Patched", (stock,), {})
    tf.apply(types.SimpleNamespace(Qwen3CoderDetector=patched))
    print(f"detector: {os.path.join(src, 'srt/function_call/qwen3_coder_detector.py')}")
    for name, text, (want_normal, want_calls), kind in CASES:
        ok, notes = True, []
        for ch in CHUNKS:
            got = run(patched, text, ch)
            base = run(stock, text, ch)
            normal_ok = want_normal is None or got[0] == want_normal
            if kind == "same":
                normal_ok = got[0] == base[0] and normal_ok
            if not (normal_ok and got[1] == want_calls):
                ok = False
                notes.append(f"chunk {ch}: got {got!r}")
            if kind == "bug" and base == got:
                notes.append(f"chunk {ch}: stock gets it right too")
            if kind == "bug-stream" and ch and base == got:
                notes.append(f"chunk {ch}: stock streaming gets it right too")
        print(f"{'PASS' if ok else 'FAIL'} {name}" + ("" if ok else "\n     " + "\n     ".join(notes[:3])))
        if ok and notes:
            print("     (" + "; ".join(notes[:2]) + ")")
        fails += not ok
    bug_text = CASES[0][1]
    print(f"     stock, not streamed, on the first case: {run(stock, bug_text, 0)!r}")
    print("all checks passed" if not fails else f"{fails} case(s) failed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
