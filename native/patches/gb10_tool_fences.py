"""Tool-call markers inside a fenced code block stay text (qwen3_coder parser).

When the model writes its tool-call syntax as an example, inside a ```xml
block (documenting the format, or showing a user what a call looks like),
SGLang's Qwen3CoderDetector (--tool-call-parser qwen3_coder, also step3p5
and nanbeige) takes it for a real call:

  * not streamed: the example is parsed as a call, and the content keeps only
    the text before the first <tool_call>, so the rest of the answer is gone;
  * streamed: the same, and any <function=NAME> in prose, even mid-line,
    opens a call.

blazux/qwen3.8-Flash-DGX fixed the same in vLLM's parser (their patch 13).
This does it for SGLang, on the text before the detector sees it:

  * inside a fenced code block, the six markers (<tool_call>, </tool_call>,
    <function=, </function>, <parameter=, </parameter>) stay text. A fence
    follows CommonMark: a run of three or more backticks or tildes at the
    start of a line (at most three spaces of indent) opens it; a run of the
    same character at least as long, with only whitespace after it, closes it;
  * outside a call, a <function= header opens one only at the start of a line.

Text inside a real call is not scanned (a parameter that holds a README with
fences is the call's business). Not covered, as in blazux's: a <tool_call>
example written outside any fence still opens a call, and a fence the model
never closes keeps the guard on for the rest of the reply. Calls the model
means are parsed as before: the guard only hides markers from the detector
(their "<" becomes U+E000 on the way in and is put back in the text that
comes out). Enabled by GB10_TOOL_FENCES=1 (scripts/serve-sglang.sh sets it
for every profile: TOOL_FENCES in ../serve.sh); patches/test_gb10_tool_fences.py
runs it against the real detector source. Written against SGLang 0.5.21; 0.5.20
and 0.5.22 have the same detector. On another one whose detector differs it
logs a warning and leaves the detector as it is, rather than stop the server.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("sglang.srt.function_call.gb10_tool_fences")

TARGET_MODULE = "sglang.srt.function_call.qwen3_coder_detector"
MARKERS = ("<tool_call>", "</tool_call>", "<function=", "</function>", "<parameter=", "</parameter>")
HIDDEN = ""  # stands for the "<" of a marker the detector must not see


def unhide(s):
    return s.replace(HIDDEN, "<") if s and HIDDEN in s else s


class Guard:
    """Fence and call state over the text of one reply, fed in pieces."""

    def __init__(self):
        self.call = None   # None, "tool" (<tool_call> .. </tool_call>) or "bare" (<function= .. </function>)
        self.fence = None  # (character, run length) of the open fence
        self.line = ""     # the current line of text outside calls, so far
        self.held = ""     # a tail that may be the start of a marker, kept for the next piece

    def _end_line(self):
        line, self.line = self.line, ""
        body = line.lstrip(" ")
        if len(line) - len(body) > 3 or body[:3] not in ("```", "~~~"):
            return
        ch = body[0]
        run = len(body) - len(body.lstrip(ch))
        rest = body[run:]
        if self.fence is None:
            if ch != "`" or "`" not in rest:  # a backtick fence's info string has no backticks
                self.fence = (ch, run)
        elif ch == self.fence[0] and run >= self.fence[1] and not rest.strip():
            self.fence = None

    def _text(self, seg):
        parts = seg.split("\n")
        self.line += parts[0]
        for p in parts[1:]:
            self._end_line()
            self.line = p

    def feed(self, text, final=False):
        """`text` with the markers that must stay text hidden. Unless `final`,
        a tail that could still become a marker is held for the next piece."""
        s, self.held = self.held + text, ""
        out, i = [], 0
        while i < len(s):
            j = s.find("<", i)
            if j < 0:
                j = len(s)
            if j > i:
                seg = s[i:j]
                out.append(seg)
                if self.call is None:
                    self._text(seg)
                i = j
                if i == len(s):
                    break
            rest = s[i:]
            m = next((k for k in MARKERS if rest.startswith(k)), None)
            if m is None:
                if not final and len(rest) < 12 and any(k.startswith(rest) for k in MARKERS):
                    self.held = rest
                    break
                out.append("<")
                if self.call is None:
                    self._text("<")
                i += 1
                continue
            i += len(m)
            if self.call is not None:  # inside a real call: the detector's business
                out.append(m)
                if (self.call, m) in (("tool", "</tool_call>"), ("bare", "</function>")):
                    self.call, self.line = None, ""
                continue
            if self.fence is not None or (m == "<function=" and self.line.strip()):
                out.append(HIDDEN + m[1:])  # an example, or a header mid-line: text
                self._text(m)
            elif m == "<tool_call>":
                self.call = "tool"
                out.append(m)
            elif m == "<function=":
                self.call = "bare"
                out.append(m)
            else:  # a stray closer or parameter outside any call: as the detector always had it
                out.append(m)
                self._text(m)
        return "".join(out)


def _unhide_result(res):
    res.normal_text = unhide(res.normal_text)
    for c in res.calls or ():
        c.parameters = unhide(c.parameters)
        if getattr(c, "name", None):
            c.name = unhide(c.name)
    return res


def apply(mod) -> None:
    cls = getattr(mod, "Qwen3CoderDetector", None)
    try:
        d = cls()
        tokens = (d.tool_call_start_token, d.tool_call_end_token, d.tool_call_prefix, d.function_end_token,
                  d.parameter_prefix, d.parameter_end_token)
        known = tokens == MARKERS and all(hasattr(d, a) for a in ("_buffer", "parsed_pos", "is_inside_tool_call")) \
            and all(callable(getattr(cls, f, None))
                    for f in ("detect_and_parse", "parse_streaming_increment", "finish"))
    except Exception:  # noqa: BLE001 - any other shape: leave it alone
        known = False
    if not known:
        logger.warning("GB10_TOOL_FENCES: this SGLang's Qwen3CoderDetector is not the one the patch was written "
                       "for (0.5.20-0.5.22); left as it is: tool-call markers in code blocks are parsed as calls")
        return
    if getattr(cls, "_gb10_tool_fences", False):
        return
    detect, increment, finish = cls.detect_and_parse, cls.parse_streaming_increment, cls.finish

    def has_tool_call(self, text):
        return self.tool_call_start_token in Guard().feed(text, final=True)

    def detect_and_parse(self, text, tools):
        masked = Guard().feed(text, final=True)
        if masked == text:
            return detect(self, text, tools)
        return _unhide_result(detect(self, masked, tools))

    def parse_streaming_increment(self, new_text, tools):
        g = self.__dict__.setdefault("_gb10_guard", Guard())
        return _unhide_result(increment(self, g.feed(new_text), tools))

    def finish_(self, tools):
        g = self.__dict__.get("_gb10_guard")
        tail = g.feed("", final=True) if g is not None else ""
        first = increment(self, tail, tools) if tail else None
        # Whatever the detector still holds outside a call (a lone "<" at the very end, which it keeps
        # waiting for a tag) is text: the stock detector drops it.
        left = self._buffer[self.parsed_pos:]
        if left and not self.is_inside_tool_call:
            self._buffer, self.parsed_pos = "", 0
        else:
            left = ""
        last = finish(self, tools)
        last.normal_text = (first.normal_text if first else "") + left + (last.normal_text or "")
        if first is not None:
            last.calls = list(first.calls or []) + list(last.calls or [])
        return _unhide_result(last)

    cls.has_tool_call = has_tool_call
    cls.detect_and_parse = detect_and_parse
    cls.parse_streaming_increment = parse_streaming_increment
    cls.finish = finish_
    cls._gb10_tool_fences = True
    logger.info("tool-call fences: markers inside fenced code blocks stay text (GB10_TOOL_FENCES)")
