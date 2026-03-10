"""
OpenAI-compatible proxy — routes /v1/* requests to the correct llamacpp sub-container.
"""

import json
import logging
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from starlette.responses import StreamingResponse

from service.containers.proxy import get_client

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
        models = registry.list_models()
        if not models:
            raise HTTPException(503, "No models registered")
        model_name = models[0]["name"]

    url = registry.get_model_url(model_name, manager)
    if not url:
        available = [m["name"] for m in registry.list_models()]
        raise HTTPException(404, f"Model '{model_name}' not found. Available: {available}")
    return url


async def _proxy_to_model(request: Request, model_name: Optional[str], path: str):
    """Proxy request to the correct model sub-container."""
    target_url = await _resolve_model_url(request, model_name)
    url = f"{target_url}{path}"

    body = await request.body()
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length", "transfer-encoding", "connection")
    }

    client = get_client()
    is_stream = b'"stream":true' in body or b'"stream": true' in body

    if is_stream:
        req = client.build_request("POST", url, content=body, headers=headers)
        response = await client.send(req, stream=True)

        if response.status_code != 200:
            error_body = await response.aread()
            await response.aclose()
            raise HTTPException(response.status_code, error_body.decode())

        async def stream_body():
            try:
                async for chunk in response.aiter_raw():
                    yield chunk
            finally:
                await response.aclose()

        return StreamingResponse(
            stream_body(),
            status_code=response.status_code,
            media_type=response.headers.get("content-type", "text/event-stream"),
        )
    else:
        resp = await client.post(url, content=body, headers=headers)
        if resp.status_code != 200:
            raise HTTPException(resp.status_code, resp.text)
        return resp.json()


@router.post("/chat/completions")
async def chat_completions(request: Request):
    body = await request.body()
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid JSON body")

    model_name = data.get("model")

    # Reconstruct request with consumed body
    scope = request.scope.copy()

    async def receive():
        return {"type": "http.request", "body": body}

    proxy_request = Request(scope, receive)
    target_url = await _resolve_model_url(request, model_name)
    return await _proxy_to_model(proxy_request, model_name, "/v1/chat/completions")


@router.post("/completions")
async def completions(request: Request):
    body = await request.body()
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid JSON body")

    model_name = data.get("model")

    async def receive():
        return {"type": "http.request", "body": body}

    proxy_request = Request(request.scope.copy(), receive)
    return await _proxy_to_model(proxy_request, model_name, "/v1/completions")


@router.post("/embeddings")
async def embeddings(request: Request):
    body = await request.body()
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid JSON body")

    model_name = data.get("model")

    async def receive():
        return {"type": "http.request", "body": body}

    proxy_request = Request(request.scope.copy(), receive)
    return await _proxy_to_model(proxy_request, model_name, "/v1/embeddings")


@router.get("/models")
async def list_models(request: Request):
    """List all available models across sub-containers."""
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
