"""
OpenAI-compatible proxy — routes /v1/* requests to the correct llamacpp sub-container.
"""

import logging
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Request
from starlette.responses import StreamingResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["openai-proxy"])


def _get_registry(request: Request):
    registry = getattr(request.app.state, "model_registry", None)
    if not registry:
        raise HTTPException(503, "Model registry not initialized")
    return registry


def _get_manager(request: Request):
    return getattr(request.app.state, "container_manager", None)


async def _resolve_model_url(request: Request, model_name: Optional[str] = None) -> str:
    """Resolve model name to sub-container URL."""
    registry = _get_registry(request)
    manager = _get_manager(request)

    if not model_name:
        # Use first available model
        models = registry.list_models()
        if not models:
            raise HTTPException(503, "No models registered")
        model_name = models[0]["name"]

    url = registry.get_model_url(model_name, manager)
    if not url:
        available = [m["name"] for m in registry.list_models()]
        raise HTTPException(404, f"Model '{model_name}' not found. Available: {available}")
    return url


async def _proxy_request(request: Request, target_url: str, path: str):
    """Proxy an HTTP request to a sub-container."""
    body = await request.body()
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length", "transfer-encoding")
    }

    url = f"{target_url}{path}"
    is_stream = b'"stream":true' in body or b'"stream": true' in body

    async with httpx.AsyncClient(timeout=300.0) as client:
        if is_stream:
            async with client.stream(
                request.method, url, content=body, headers=headers
            ) as resp:
                if resp.status_code != 200:
                    error_body = await resp.aread()
                    raise HTTPException(resp.status_code, error_body.decode())

                async def stream_gen():
                    async for chunk in resp.aiter_bytes():
                        yield chunk

                return StreamingResponse(
                    stream_gen(),
                    status_code=resp.status_code,
                    media_type=resp.headers.get("content-type", "text/event-stream"),
                )
        else:
            resp = await client.request(
                request.method, url, content=body, headers=headers
            )
            return resp.json() if resp.status_code == 200 else HTTPException(resp.status_code, resp.text)


@router.post("/chat/completions")
async def chat_completions(request: Request):
    import json
    body = await request.body()
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid JSON body")

    model_name = data.get("model")
    target_url = await _resolve_model_url(request, model_name)
    # Re-create request with body (already consumed)
    from starlette.requests import Request as StarletteRequest
    scope = request.scope.copy()

    async def receive():
        return {"type": "http.request", "body": body}

    proxy_request = StarletteRequest(scope, receive)
    return await _proxy_request(proxy_request, target_url, "/v1/chat/completions")


@router.post("/completions")
async def completions(request: Request):
    import json
    body = await request.body()
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid JSON body")

    model_name = data.get("model")
    target_url = await _resolve_model_url(request, model_name)

    async def receive():
        return {"type": "http.request", "body": body}

    proxy_request = Request(request.scope.copy(), receive)
    return await _proxy_request(proxy_request, target_url, "/v1/completions")


@router.post("/embeddings")
async def embeddings(request: Request):
    import json
    body = await request.body()
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid JSON body")

    model_name = data.get("model")
    target_url = await _resolve_model_url(request, model_name)

    async def receive():
        return {"type": "http.request", "body": body}

    proxy_request = Request(request.scope.copy(), receive)
    return await _proxy_request(proxy_request, target_url, "/v1/embeddings")


@router.get("/models")
async def list_models(request: Request):
    """List all available models across sub-containers."""
    import time
    registry = _get_registry(request)
    models = registry.list_models()

    data = []
    for m in models:
        data.append({
            "id": m["name"],
            "object": "model",
            "created": int(time.time()),
            "owned_by": "llamacpp-router-agentified",
            "permission": [],
            "root": m["name"],
            "parent": None,
        })
    return {"object": "list", "data": data}
