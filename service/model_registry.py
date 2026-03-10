"""
llamacpp-router-agentified — Model Registry

Parses MODELS env var, reads model catalog configs, and generates
ContainerDefinition entries for each model. Manages the mapping
from model name → sub-container.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from service.containers.models import ContainerDefinition, HealthCheckConfig, ResourceConfig

logger = logging.getLogger(__name__)


class ModelEntry:
    """A model in the registry."""

    def __init__(self, name: str, catalog_config: dict, container_name: str):
        self.name = name
        self.catalog = catalog_config
        self.container_name = container_name

    @property
    def model_type(self) -> str:
        return self.catalog.get("type", "chat")

    @property
    def is_embedding(self) -> bool:
        return self.model_type == "embedding"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.model_type,
            "container": self.container_name,
            "description": self.catalog.get("description", ""),
        }


class ModelRegistry:
    """
    Registry of models. Each model maps to a llamacpp-agentified sub-container.

    Reads MODELS env var (comma-separated model names) and generates
    container definitions for the ContainerManager.
    """

    def __init__(self, config_dir: str = "/app/config"):
        self.config_dir = config_dir
        self.catalog_dir = Path(config_dir) / "models"
        self.models: Dict[str, ModelEntry] = {}
        self._llamacpp_image = os.getenv("LLAMACPP_IMAGE", "llamacpp-agentified:latest")
        self._llamacpp_data_volume = os.getenv("LLAMACPP_DATA_VOLUME", "llamacpp-shared-models")
        self._network_name = os.getenv("COMPOSE_PROJECT_NAME", "llamacpp-router") + "-network"
        self._gpu_per_model = float(os.getenv("GPU_FRACTION_PER_MODEL", "0.0"))

    def load_models_from_env(self) -> List[str]:
        """Parse MODELS env var, register each model."""
        models_str = os.getenv("MODELS", "")
        if not models_str:
            logger.warning("MODELS env var not set — no models will be loaded")
            return []

        model_names = [m.strip() for m in models_str.split(",") if m.strip()]
        loaded = []

        for name in model_names:
            try:
                self._register_model(name)
                loaded.append(name)
            except Exception as e:
                logger.error(f"Failed to register model '{name}': {e}")

        logger.info(f"Registered {len(loaded)} models: {loaded}")
        return loaded

    def _register_model(self, name: str):
        """Register a model from catalog config."""
        catalog_path = self.catalog_dir / f"{name}.json"
        if not catalog_path.exists():
            available = [p.stem for p in self.catalog_dir.glob("*.json")]
            raise FileNotFoundError(
                f"No catalog config for '{name}'. Available: {available}"
            )

        with open(catalog_path) as f:
            catalog = json.load(f)

        container_name = f"llm-{name}"
        self.models[name] = ModelEntry(name, catalog, container_name)

    def get_model(self, name: str) -> Optional[ModelEntry]:
        return self.models.get(name)

    def list_models(self) -> List[dict]:
        return [m.to_dict() for m in self.models.values()]

    def generate_container_definitions(self) -> dict[str, ContainerDefinition]:
        """
        Generate ContainerDefinition objects for ContainerManager.
        Each model becomes a llamacpp-agentified sub-container.
        """
        definitions = {}

        for name, entry in self.models.items():
            container_name = entry.container_name
            catalog = entry.catalog

            # Environment for the llamacpp-agentified container
            env = {
                "MODEL_NAME": name,
                "MODEL_DIR": "/data/models",
                "SERVICE_PORT": "8080",
                "GPU_LAYERS": str(catalog.get("gpu_layers", -1)),
            }

            # Add HF_TOKEN if set in router
            hf_token = os.getenv("HF_TOKEN", "")
            if hf_token:
                env["HF_TOKEN"] = hf_token

            # GPU reservation
            if self._gpu_per_model > 0:
                env["GPU_FRACTION"] = str(self._gpu_per_model)

            definitions[container_name] = ContainerDefinition(
                image=self._llamacpp_image,
                internal_port=8080,
                environment=env,
                volumes={self._llamacpp_data_volume: "/data"},
                health_check=HealthCheckConfig(
                    endpoint="/health",
                    interval_seconds=15,
                    timeout_seconds=5,
                    retries=3,
                ),
                resources=ResourceConfig(cpu_limit="4", memory_limit="8g"),
                idle_timeout_seconds=600,
                startup_timeout_seconds=600,  # model download can take a while
                auto_start=True,
                group="inference",
            )

        return definitions

    def get_model_url(self, name: str, container_manager=None) -> Optional[str]:
        """Get the internal URL for a model's container."""
        entry = self.models.get(name)
        if not entry:
            return None

        if container_manager:
            url = container_manager.resolve_url(entry.container_name)
            if url:
                return url

        # Fallback: assume container name resolves via Docker DNS
        return f"http://{entry.container_name}:8080"
