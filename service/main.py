"""
llamacpp-router-agentified — Main FastAPI Application

Ollama-like router that manages multiple llamacpp-agentified sub-containers.
Routes /v1/* (OpenAI) and /api/* (Ollama) requests to the correct model container.
"""

import json
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from service.config import get_config
from service.routers import health, containers as containers_router


def _setup_logging(config):
    level = getattr(logging, config.log_level.upper(), logging.INFO)
    if config.log_format == "json":
        fmt = '{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}'
    else:
        fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt, stream=sys.stdout, force=True)


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = get_config()
    _setup_logging(config)
    logger.info(f"Starting {config.name} v{config.version}")

    # --- STARTUP ---

    # 1. Initialize model registry from MODELS env var
    from service.model_registry import ModelRegistry
    registry = ModelRegistry(config.config_dir)
    loaded_models = registry.load_models_from_env()
    app.state.model_registry = registry

    # 2. Container manager — merge static service.json definitions with
    #    dynamically generated model container definitions
    container_manager = None
    from service.containers.manager import ContainerManager, load_container_definitions

    # Load static definitions from service.json (if any)
    static_definitions = {}
    static_defaults = {}
    json_path = Path(config.config_dir) / "service.json"
    if json_path.exists():
        try:
            with open(json_path) as f:
                config_data = json.load(f)
            if config_data.get("containers", {}).get("definitions"):
                static_definitions, static_defaults = load_container_definitions(config_data)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"service.json load: {e}")

    # Generate model container definitions
    model_definitions = registry.generate_container_definitions()

    # Merge: model definitions + static definitions
    all_definitions = {**static_definitions, **model_definitions}

    if all_definitions:
        try:
            container_manager = ContainerManager(all_definitions, static_defaults)
            app.state.container_manager = container_manager
            await container_manager.start_background_tasks()
            logger.info(
                f"Container manager: {len(all_definitions)} containers "
                f"({len(model_definitions)} model, {len(static_definitions)} static)"
            )
        except Exception as e:
            logger.error(f"Container manager init failed: {e}")

    # 3. Mount MCP server if enabled
    if config.mcp_enabled:
        from mcp.sse_server import setup_mcp_server
        setup_mcp_server(app)
        logger.info(f"MCP SSE at {config.mcp_path_prefix}/sse")

    logger.info(f"Ready — serving {len(loaded_models)} models: {loaded_models}")

    yield

    # --- SHUTDOWN ---
    if container_manager:
        await container_manager.shutdown()
    logger.info(f"Shutting down {config.name}")


def create_app() -> FastAPI:
    config = get_config()

    app = FastAPI(
        title=config.name,
        version=config.version,
        description=config.description,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    origins = config.cors_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=("*" not in origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from service.routers import openai_proxy, ollama_compat

    app.include_router(health.router)
    app.include_router(openai_proxy.router)
    app.include_router(ollama_compat.router)
    app.include_router(containers_router.router)

    return app


app = create_app()
