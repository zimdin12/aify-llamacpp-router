"""
Ollama-compatible API endpoints.

Translates Ollama API format (/api/chat, /api/generate, /api/embeddings)
to OpenAI format and proxies to the correct llamacpp sub-container.
"""

import json
import logging
import time
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from service.containers.proxy import get_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["ollama-compat"])


# ---------------------------------------------------------------------------
# Ollama request models
# ---------------------------------------------------------------------------

class OllamaMessage(BaseModel):
    role: str
    content: str


class OllamaChatRequest(BaseModel):
    model: str
    messages: List[OllamaMessage]
    stream: bool = True
    options: Optional[dict] = None


class OllamaGenerateRequest(BaseModel):
    model: str
    prompt: str
    stream: bool = True
    options: Optional[dict] = None


class OllamaEmbeddingRequest(BaseModel):
    model: str
    prompt: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_registry(request: Request):
    registry = getattr(request.app.state, "model_registry", None)
    if not registry:
        raise HTTPException(503, "Model registry not initialized")
    return registry


def _get_manager(request: Request):
    return getattr(request.app.state, "container_manager", None)


def _resolve_url(request: Request, model_name: str) -> str:
    registry = _get_registry(request)
    manager = _get_manager(request)
    url = registry.get_model_url(model_name, manager)
    if not url:
        available = [m["name"] for m in registry.list_models()]
        raise HTTPException(404, f"Model '{model_name}' not found. Available: {available}")
    return url


def _extract_options(options: Optional[dict]) -> dict:
    """Convert Ollama options to OpenAI params."""
    if not options:
        return {}
    params = {}
    if "temperature" in options:
        params["temperature"] = options["temperature"]
    if "top_p" in options:
        params["top_p"] = options["top_p"]
    if "num_predict" in options:
        params["max_tokens"] = options["num_predict"]
    if "stop" in options:
        params["stop"] = options["stop"]
    if "frequency_penalty" in options:
        params["frequency_penalty"] = options["frequency_penalty"]
    if "presence_penalty" in options:
        params["presence_penalty"] = options["presence_penalty"]
    return params


# ---------------------------------------------------------------------------
# /api/chat
# ---------------------------------------------------------------------------

@router.post("/chat")
async def chat(body: OllamaChatRequest, request: Request):
    target_url = _resolve_url(request, body.model)

    # Translate to OpenAI format
    openai_body = {
        "model": body.model,
        "messages": [{"role": m.role, "content": m.content} for m in body.messages],
        "stream": body.stream,
        **_extract_options(body.options),
    }

    client = get_client()

    if body.stream:
        req = client.build_request(
            "POST", f"{target_url}/v1/chat/completions", json=openai_body,
        )
        response = await client.send(req, stream=True)

        if response.status_code != 200:
            error = await response.aread()
            await response.aclose()
            raise HTTPException(response.status_code, error.decode())

        async def ollama_stream():
            try:
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        yield json.dumps({
                            "model": body.model,
                            "done": True,
                            "message": {"role": "assistant", "content": ""},
                        }) + "\n"
                        return

                    try:
                        chunk = json.loads(data)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        yield json.dumps({
                            "model": body.model,
                            "done": False,
                            "message": {"role": "assistant", "content": content},
                        }) + "\n"
                    except (json.JSONDecodeError, IndexError, KeyError):
                        continue
            finally:
                await response.aclose()

        return StreamingResponse(ollama_stream(), media_type="application/x-ndjson")

    else:
        resp = await client.post(
            f"{target_url}/v1/chat/completions", json=openai_body,
        )
        if resp.status_code != 200:
            raise HTTPException(resp.status_code, resp.text)

        result = resp.json()
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = result.get("usage", {})

        return {
            "model": body.model,
            "done": True,
            "message": {"role": "assistant", "content": content},
            "eval_count": usage.get("completion_tokens", 0),
            "prompt_eval_count": usage.get("prompt_tokens", 0),
        }


# ---------------------------------------------------------------------------
# /api/generate
# ---------------------------------------------------------------------------

@router.post("/generate")
async def generate(body: OllamaGenerateRequest, request: Request):
    target_url = _resolve_url(request, body.model)

    openai_body = {
        "model": body.model,
        "prompt": body.prompt,
        "stream": body.stream,
        **_extract_options(body.options),
    }

    client = get_client()

    if body.stream:
        req = client.build_request(
            "POST", f"{target_url}/v1/completions", json=openai_body,
        )
        response = await client.send(req, stream=True)

        if response.status_code != 200:
            error = await response.aread()
            await response.aclose()
            raise HTTPException(response.status_code, error.decode())

        async def ollama_stream():
            try:
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        yield json.dumps({"model": body.model, "done": True, "response": ""}) + "\n"
                        return
                    try:
                        chunk = json.loads(data)
                        text = chunk.get("choices", [{}])[0].get("text", "")
                        yield json.dumps({"model": body.model, "done": False, "response": text}) + "\n"
                    except (json.JSONDecodeError, IndexError, KeyError):
                        continue
            finally:
                await response.aclose()

        return StreamingResponse(ollama_stream(), media_type="application/x-ndjson")

    else:
        resp = await client.post(f"{target_url}/v1/completions", json=openai_body)
        if resp.status_code != 200:
            raise HTTPException(resp.status_code, resp.text)

        result = resp.json()
        text = result.get("choices", [{}])[0].get("text", "")

        return {
            "model": body.model,
            "done": True,
            "response": text,
        }


# ---------------------------------------------------------------------------
# /api/embeddings
# ---------------------------------------------------------------------------

@router.post("/embeddings")
async def embeddings_ollama(body: OllamaEmbeddingRequest, request: Request):
    target_url = _resolve_url(request, body.model)

    openai_body = {"model": body.model, "input": body.prompt}

    client = get_client()
    resp = await client.post(f"{target_url}/v1/embeddings", json=openai_body)
    if resp.status_code != 200:
        raise HTTPException(resp.status_code, resp.text)

    result = resp.json()
    embedding = result.get("data", [{}])[0].get("embedding", [])

    return {"embedding": embedding}


# ---------------------------------------------------------------------------
# /api/tags  (list models — Ollama format)
# ---------------------------------------------------------------------------

@router.get("/tags")
async def tags(request: Request):
    registry = _get_registry(request)
    models = registry.list_models()

    return {
        "models": [
            {
                "name": m["name"],
                "model": m["name"],
                "modified_at": "",
                "size": 0,
                "digest": "",
                "details": {
                    "format": "gguf",
                    "family": m.get("description", ""),
                    "parameter_size": "",
                    "quantization_level": "",
                },
            }
            for m in models
        ]
    }


# ---------------------------------------------------------------------------
# /api/show  (model info — Ollama format)
# ---------------------------------------------------------------------------

@router.post("/show")
async def show(request: Request):
    body = await request.json()
    model_name = body.get("name", body.get("model", ""))

    registry = _get_registry(request)
    entry = registry.get_model(model_name)
    if not entry:
        raise HTTPException(404, f"Model '{model_name}' not found")

    return {
        "modelfile": "",
        "parameters": "",
        "template": "",
        "details": {
            "format": "gguf",
            "family": entry.catalog.get("description", ""),
            "parameter_size": "",
            "quantization_level": "",
        },
        "model_info": entry.catalog,
    }
