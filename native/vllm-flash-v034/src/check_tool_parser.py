#!/usr/bin/env python3
"""Build-time check of the qwen3 tool-call parser (the Dockerfile runs it after the two
patches in src/patches/): the cases ../bench/consistency.py section 1 and blazux's tests
cover, fed through vLLM's own StreamingParserEngine with the qwen3 configuration that
--tool-call-parser qwen3_xml runs, in chunks of 1, 7 and all characters.

  1. a call shown as an example inside a ```xml fence stays text;
  2. a <function=NAME> header in the middle of a line stays text;
  3. a real call still parses (name and argument);
  4. a quoted <tool_call> in the reasoning keeps the answer after </think>.

    python3 check_tool_parser.py        exit 0 when all hold, 1 with the failures listed
"""
import sys

from vllm.parser.engine.events import EventType
from vllm.parser.engine.streaming_parser_engine import StreamingParserEngine
from vllm.parser.qwen3 import qwen3_config

CALL = ("<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n"
        "</function>\n</tool_call>")


def parse(text, chunk, thinking):
    engine = StreamingParserEngine(qwen3_config(thinking=thinking), None)
    events = []
    for i in range(0, len(text), chunk):
        events.extend(engine.feed(text[i:i + chunk], []))
    events.extend(engine.finish())
    return events


def joined(events, kind):
    return "".join(e.value or "" for e in events if e.type == kind)


def calls(events):
    return sum(e.type == EventType.TOOL_CALL_START for e in events)


def cases():
    fenced = "Here is how a call looks:\n```xml\n" + CALL.replace("Paris", "Berlin") + "\n```\nDONE"
    midline = "In this format the header <function=NAME> opens a call. DONE"
    quoted = "The literal marker is `<tool_call>` in prose."
    for chunk in (1, 7, 10**6):
        ev = parse(fenced, chunk, thinking=False)
        yield (f"fenced example stays text (chunk {chunk})",
               calls(ev) == 0 and joined(ev, EventType.TEXT_CHUNK) == fenced)
        ev = parse(midline, chunk, thinking=False)
        yield (f"mid-line <function=NAME> stays text (chunk {chunk})",
               calls(ev) == 0 and joined(ev, EventType.TEXT_CHUNK) == midline)
        ev = parse("Calling it.\n" + CALL, chunk, thinking=False)
        yield (f"a real call parses (chunk {chunk})",
               calls(ev) == 1 and joined(ev, EventType.TOOL_NAME) == "get_weather"
               and "Paris" in joined(ev, EventType.ARG_VALUE_CHUNK))
        ev = parse(quoted + "</think>The answer is 42.", chunk, thinking=True)
        yield (f"quoted <tool_call> in reasoning keeps the answer (chunk {chunk})",
               calls(ev) == 0 and joined(ev, EventType.REASONING_CHUNK) == quoted
               and joined(ev, EventType.TEXT_CHUNK) == "The answer is 42.")


def main():
    failed = [name for name, ok in cases() if not ok]
    for name in failed:
        print("FAIL", name)
    print("tool-call parser check:", "all cases hold" if not failed else f"{len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
