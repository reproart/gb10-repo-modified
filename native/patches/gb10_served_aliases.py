"""More model ids for one SGLang server: `/v1/models` lists the profile's own
name first and then each alias, `/v1/models/<alias>` answers, and the Responses
API (the one endpoint that checks the request's model) accepts them.

vLLM takes several --served-model-name values (vllm-flash-ar serves
"qwen3.8-flash-next qwen"); SGLang 0.5.21 takes one. Its chat, completions and
Anthropic endpoints never check the request's `model` (a reply echoes it back),
so a client that asks for "qwen" already gets an answer; what it does not get is
"qwen" in the model list a client fetches, and the Responses API answers 404.
With a stable alias the client keeps its settings whichever profile serves.

GB10_SERVED_ALIASES: space-separated ids (../serve.sh: SERVED_ALIASES, default
"qwen"). Written against SGLang 0.5.21.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("sglang.srt.entrypoints.gb10_served_aliases")

TARGET_MODULE = "sglang.srt.entrypoints.http_server"
RESPONSES_MODULE = "sglang.srt.entrypoints.openai.serving_responses"
MODELS_PATH = "/v1/models"
MODEL_PATH = "/v1/models/{model:path}"


def aliases() -> list:
    out = []
    for name in os.environ.get("GB10_SERVED_ALIASES", "").split():
        if name not in out:
            out.append(name)
    return out


def _replace_get_route(app, path, endpoint) -> bool:
    from fastapi.routing import APIRoute

    for i, route in enumerate(app.router.routes):
        if isinstance(route, APIRoute) and route.path == path and "GET" in route.methods:
            app.router.routes[i] = APIRoute(
                path, endpoint, methods=["GET"], response_class=route.response_class,
                dependencies=route.dependencies, name=route.name,
                dependency_overrides_provider=route.dependency_overrides_provider)
            return True
    return False


def apply_http(mod) -> None:
    names = aliases()
    if not names or getattr(mod, "_gb10_served_aliases", None):
        return
    orig_list, orig_get = mod.available_models, mod.retrieve_model
    from sglang.srt.entrypoints.openai.protocol import ModelCard

    def state():
        tm = mod._global_state.tokenizer_manager
        return tm.served_model_name, tm.model_config.context_len

    async def available_models():
        listing = await orig_list()
        served, ctx = state()
        have = {card.id for card in listing.data}
        at = 1 if listing.data else 0
        for name in names:
            if name not in have:
                listing.data.insert(at, ModelCard(id=name, root=served, max_model_len=ctx))
                at += 1
        return listing

    async def retrieve_model(model: str):
        served, ctx = state()
        if model in names and model != served:
            return ModelCard(id=model, root=served, max_model_len=ctx)
        return await orig_get(model)

    done = [_replace_get_route(mod.app, MODELS_PATH, available_models),
            _replace_get_route(mod.app, MODEL_PATH, retrieve_model)]
    if not all(done):
        logger.warning("GB10_SERVED_ALIASES: %s not found in SGLang's routes; aliases not listed",
                       " / ".join(p for p, ok in zip((MODELS_PATH, MODEL_PATH), done, strict=True) if not ok))
    mod._gb10_served_aliases = names
    logger.info("GB10_SERVED_ALIASES: also listed and answered as %s (patches/gb10_served_aliases.py)",
                ", ".join(names))


def apply_responses(mod) -> None:
    names = aliases()
    cls = getattr(mod, "OpenAIServingResponses", None)
    if not names or cls is None or getattr(cls, "_gb10_served_aliases", None):
        return
    orig = cls._known_model_names

    def _known_model_names(self):
        return set(orig(self)) | set(names)

    cls._known_model_names = _known_model_names
    cls._gb10_served_aliases = names
