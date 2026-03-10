"""
Health and service info endpoints for llamacpp-router-agentified.
"""

from fastapi import APIRouter, Request
from service.config import get_config

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    return {"status": "healthy"}


@router.get("/ready")
async def ready(request: Request):
    checks = {}

    registry = getattr(request.app.state, "model_registry", None)
    if registry:
        checks["models"] = [m["name"] for m in registry.list_models()]
        checks["model_count"] = len(checks["models"])

    manager = getattr(request.app.state, "container_manager", None)
    if manager is not None:
        checks["container_manager"] = "initialized"
        checks["docker"] = "connected" if manager.docker else "unavailable"

    return {"status": "ready", "checks": checks}


@router.get("/info")
async def info(request: Request):
    config = get_config()
    host = request.headers.get("host", f"localhost:{config.port}")
    base = f"http://{host}"

    registry = getattr(request.app.state, "model_registry", None)
    models = registry.list_models() if registry else []

    response = {
        "name": config.name,
        "version": config.version,
        "description": config.description,
        "models": models,
        "endpoints": {
            "openai_chat": f"{base}/v1/chat/completions",
            "openai_completions": f"{base}/v1/completions",
            "openai_embeddings": f"{base}/v1/embeddings",
            "openai_models": f"{base}/v1/models",
            "ollama_chat": f"{base}/api/chat",
            "ollama_generate": f"{base}/api/generate",
            "ollama_tags": f"{base}/api/tags",
            "health": f"{base}/health",
            "docs": f"{base}/docs",
        },
    }

    manager = getattr(request.app.state, "container_manager", None)
    if manager is not None:
        response["containers"] = manager.list_containers()

    return response
