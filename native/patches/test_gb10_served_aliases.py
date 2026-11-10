#!/usr/bin/env python3
"""Tests for gb10_served_aliases (CPU; needs fastapi + httpx, which SGLang's venv has):

    python3 patches/test_gb10_served_aliases.py

A stand-in for sglang.srt.entrypoints.http_server with SGLang 0.5.21's two model
routes (and one route the patch must leave alone) goes through the patch and a
FastAPI TestClient: the list carries the served name first and then the aliases,
an alias resolves to the served model, an unknown id is still a 404, and the
Responses API's known names grow by the aliases. Without GB10_SERVED_ALIASES
nothing changes.
"""

import os
import sys
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import fastapi
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover
    fastapi = None

import gb10_served_aliases as sa  # noqa: E402


def protocol_module():
    try:
        from sglang.srt.entrypoints.openai import (
            protocol,  # the real one, in SGLang's venv
        )
        return protocol
    except ImportError:
        pass
    import time
    from typing import List, Optional

    from pydantic import BaseModel, Field

    class ModelCard(BaseModel):
        id: str
        object: str = "model"
        created: int = Field(default_factory=lambda: int(time.time()))
        owned_by: str = "sglang"
        root: Optional[str] = None
        parent: Optional[str] = None
        max_model_len: Optional[int] = None

    class ModelList(BaseModel):
        object: str = "list"
        data: List[ModelCard] = []

    names = ("sglang", "sglang.srt", "sglang.srt.entrypoints", "sglang.srt.entrypoints.openai")
    for name in names:
        sys.modules.setdefault(name, types.ModuleType(name))
    mod = types.ModuleType("sglang.srt.entrypoints.openai.protocol")
    mod.ModelCard, mod.ModelList = ModelCard, ModelList
    sys.modules[mod.__name__] = mod
    sys.modules["sglang.srt.entrypoints.openai"].protocol = mod
    return mod


def fake_http_server():
    """SGLang 0.5.21's routes, as http_server.py declares them."""
    from fastapi.responses import JSONResponse

    protocol = protocol_module()
    ModelCard, ModelList = protocol.ModelCard, protocol.ModelList
    mod = types.ModuleType("fake_http_server")
    app = fastapi.FastAPI()
    tm = types.SimpleNamespace(served_model_name="qwen3.8-27b-sglang",
                               model_config=types.SimpleNamespace(context_len=262144))
    mod._global_state = types.SimpleNamespace(tokenizer_manager=tm)

    @app.get("/v1/models", response_class=JSONResponse)
    async def available_models():
        served = mod._global_state.tokenizer_manager.served_model_name
        return ModelList(data=[ModelCard(id=served, root=served, max_model_len=262144)])

    @app.get("/v1/models/{model:path}", response_class=JSONResponse)
    async def retrieve_model(model: str):
        if model != mod._global_state.tokenizer_manager.served_model_name:
            return JSONResponse(status_code=404, content={"error": {"code": "model_not_found"}})
        return ModelCard(id=model, root=model, max_model_len=262144)

    @app.post("/v1/chat/completions")
    async def chat(body: dict):
        return {"model": body.get("model")}

    mod.app, mod.available_models, mod.retrieve_model = app, available_models, retrieve_model
    return mod


class EnvCase(unittest.TestCase):
    def setUp(self):
        self.old = os.environ.get("GB10_SERVED_ALIASES")
        self.addCleanup(self.restore)

    def restore(self):
        if self.old is None:
            os.environ.pop("GB10_SERVED_ALIASES", None)
        else:
            os.environ["GB10_SERVED_ALIASES"] = self.old


class AliasNamesTest(EnvCase):
    def test_parse(self):
        os.environ["GB10_SERVED_ALIASES"] = " qwen  local qwen "
        self.assertEqual(sa.aliases(), ["qwen", "local"])
        os.environ["GB10_SERVED_ALIASES"] = ""
        self.assertEqual(sa.aliases(), [])


@unittest.skipIf(fastapi is None, "fastapi not installed (SGLang's venv has it)")
class RoutesTest(EnvCase):
    def test_list_get_and_404(self):
        os.environ["GB10_SERVED_ALIASES"] = "qwen local"
        mod = fake_http_server()
        n_routes = len(mod.app.router.routes)
        with self.assertLogs("sglang.srt.entrypoints.gb10_served_aliases", "INFO"):
            sa.apply_http(mod)
        sa.apply_http(mod)   # idempotent
        self.assertEqual(len(mod.app.router.routes), n_routes)
        c = TestClient(mod.app)
        ids = [m["id"] for m in c.get("/v1/models").json()["data"]]
        self.assertEqual(ids, ["qwen3.8-27b-sglang", "qwen", "local"])
        card = c.get("/v1/models/qwen").json()
        self.assertEqual((card["id"], card["root"], card["max_model_len"]), ("qwen", "qwen3.8-27b-sglang", 262144))
        self.assertEqual(c.get("/v1/models/qwen3.8-27b-sglang").status_code, 200)
        self.assertEqual(c.get("/v1/models/other").status_code, 404)
        self.assertEqual(c.post("/v1/chat/completions", json={"model": "qwen"}).json(), {"model": "qwen"})

    def test_off(self):
        os.environ.pop("GB10_SERVED_ALIASES", None)
        mod = fake_http_server()
        sa.apply_http(mod)
        c = TestClient(mod.app)
        self.assertEqual([m["id"] for m in c.get("/v1/models").json()["data"]], ["qwen3.8-27b-sglang"])
        self.assertEqual(c.get("/v1/models/qwen").status_code, 404)

    def test_served_name_is_an_alias_too(self):
        os.environ["GB10_SERVED_ALIASES"] = "qwen3.8-27b-sglang qwen"
        mod = fake_http_server()
        sa.apply_http(mod)
        ids = [m["id"] for m in TestClient(mod.app).get("/v1/models").json()["data"]]
        self.assertEqual(ids, ["qwen3.8-27b-sglang", "qwen"])


class ResponsesTest(EnvCase):
    def test_known_names(self):
        os.environ["GB10_SERVED_ALIASES"] = "qwen"

        class OpenAIServingResponses:
            def _known_model_names(self):
                return {"qwen3.8-27b-sglang"}

        mod = types.ModuleType("fake_responses")
        mod.OpenAIServingResponses = OpenAIServingResponses
        sa.apply_responses(mod)
        sa.apply_responses(mod)
        self.assertEqual(OpenAIServingResponses()._known_model_names(), {"qwen3.8-27b-sglang", "qwen"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
