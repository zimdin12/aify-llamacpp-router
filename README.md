# llamacpp-router-agentified

Ollama-like LLM router that manages multiple [llamacpp-agentified](https://github.com/zimdin12/llamacpp-agentified) sub-containers. Each model runs in its own isolated container, and the router provides a unified API with both OpenAI and Ollama compatibility.

Built on [agentify-container](https://github.com/zimdin12/agentify-container) — uses the Python Docker SDK to spawn, health-check, and auto-shutdown model containers.

## Quick Start

```bash
# 1. Build llamacpp-agentified image first (the router spawns these)
cd ../llamacpp-agentified
docker compose build
docker tag llamacpp-inference llamacpp-agentified:latest

# 2. Set up the router
cd ../llamacpp-router-agentified
cp .env.example .env
cp config/service.example.json config/service.json

# Edit .env — set MODELS to the models you want (comma-separated)
# MODELS=qwen3-4b,qwen3-embedding-0.6b

docker compose up -d --build
```

## How It Works

```
Client Request
    │
    ▼
llamacpp-router (:11434)
    ├── /v1/chat/completions  ──► routes by "model" field
    ├── /api/chat             ──► Ollama-compatible
    └── /api/tags             ──► lists all models
         │
         ▼
    ModelRegistry (parses MODELS env var)
         │
    ┌────┴────────────────┐
    ▼                     ▼
llm-qwen3-4b (:8080)   llm-qwen3-embedding-0.6b (:8080)
(llamacpp-agentified)   (llamacpp-agentified)
```

1. On startup, the router reads `MODELS` env var (e.g., `qwen3-4b,mistral-7b`)
2. For each model, it generates a container definition and registers it with the ContainerManager
3. ContainerManager spawns llamacpp-agentified containers via Docker SDK
4. Requests are routed to the correct container based on the `model` field

## API Endpoints

### OpenAI-Compatible (proxied to sub-containers)

| Endpoint | Method | Description |
|---|---|---|
| `/v1/chat/completions` | POST | Chat completion (routes by `model` field) |
| `/v1/completions` | POST | Text completion |
| `/v1/embeddings` | POST | Text embeddings |
| `/v1/models` | GET | List all available models |

### Ollama-Compatible

| Endpoint | Method | Description |
|---|---|---|
| `/api/chat` | POST | Ollama chat format (translated to OpenAI internally) |
| `/api/generate` | POST | Ollama generate format |
| `/api/embeddings` | POST | Ollama embedding format |
| `/api/tags` | GET | List models (Ollama format) |
| `/api/show` | POST | Model details (Ollama format) |

### Service

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/ready` | GET | Readiness with model list |
| `/info` | GET | Full service discovery (models, endpoints, containers) |
| `/docs` | GET | Swagger UI |
| `/api/v1/containers` | GET | Container management API (from agentify-container) |

## Usage Examples

### OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="unused")

response = client.chat.completions.create(
    model="qwen3-4b",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

### Ollama SDK

```python
import ollama

client = ollama.Client(host="http://localhost:11434")

response = client.chat(
    model="qwen3-4b",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response["message"]["content"])
```

### curl

```bash
# OpenAI format
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-4b","messages":[{"role":"user","content":"Hello"}]}'

# Ollama format
curl http://localhost:11434/api/chat \
  -d '{"model":"qwen3-4b","messages":[{"role":"user","content":"Hello"}],"stream":false}'

# List models
curl http://localhost:11434/v1/models
curl http://localhost:11434/api/tags
```

## Configuration

### Environment Variables

| Env Var | Default | Description |
|---|---|---|
| `MODELS` | *(empty)* | Comma-separated model names to serve |
| `SERVICE_PORT` | `11434` | Router API port (Ollama default) |
| `LLAMACPP_IMAGE` | `llamacpp-agentified:latest` | Docker image for model containers |
| `LLAMACPP_DATA_VOLUME` | `llamacpp-shared-models` | Shared volume for model files |
| `GPU_FRACTION_PER_MODEL` | `0.0` | GPU memory fraction per model |
| `HF_TOKEN` | *(empty)* | HuggingFace token (passed to sub-containers) |
| `MCP_ENABLED` | `true` | Enable MCP SSE server |
| `LOG_LEVEL` | `info` | Logging level |

### Model Catalog

Models are defined in `config/models/*.json`. The catalog is shared with llamacpp-agentified — same format:

```json
{
  "repo": "unsloth/Qwen3-4B-GGUF",
  "filename": "Qwen3-4B-Q4_K_M.gguf",
  "context_length": 32768,
  "gpu_layers": -1,
  "chat_format": "chatml",
  "description": "Qwen3 4B Q4_K_M",
  "type": "chat",
  "embedding_dims": null
}
```

Add new models by dropping a JSON file in `config/models/` and adding the name to `MODELS`.

### Static Containers

Beyond dynamically-generated model containers, you can define static containers in `config/service.json` (e.g., a shared embedding service, a reranker, etc.). These merge with the model containers.

## Architecture

```
llamacpp-router-agentified/
├── config/
│   ├── models/                  # Shared model catalog
│   │   ├── qwen3-4b.json
│   │   └── ...
│   └── service.example.json     # Container manager defaults
├── service/
│   ├── main.py                  # FastAPI app with model registry + container manager
│   ├── model_registry.py        # MODELS env → container definitions
│   ├── config.py                # Environment-based configuration
│   ├── containers/              # Docker SDK container management (from agentify-container)
│   │   ├── manager.py
│   │   ├── models.py
│   │   ├── proxy.py
│   │   └── gpu.py
│   └── routers/
│       ├── health.py            # /health, /ready, /info
│       ├── openai_proxy.py      # /v1/* → sub-container proxy
│       ├── ollama_compat.py     # /api/* Ollama-compatible endpoints
│       └── containers.py        # Container management API
├── mcp_local/
│   └── sse_server.py            # MCP SSE server with container + model tools
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

## Multi-Model GPU Sharing

When running multiple models on a single GPU, set `GPU_FRACTION_PER_MODEL`:

```bash
# 3 models sharing one GPU
MODELS=qwen3-4b,qwen3-1.7b,qwen3-embedding-0.6b
GPU_FRACTION_PER_MODEL=0.33
```

The container manager tracks GPU allocations and prevents overcommitment.

## MCP Integration

The router exposes MCP tools for AI agents (Claude Code, etc.):

```bash
claude mcp add llamacpp-router --transport sse http://localhost:11434/mcp/claude-code/sse
```

## Related Projects

- **[llamacpp-agentified](https://github.com/zimdin12/llamacpp-agentified)** — The model container this router manages
- **[openmemory-agentified](https://github.com/zimdin12/openmemory-agentified)** — Hybrid memory system that uses this as LLM backend
- **[agentify-container](https://github.com/zimdin12/agentify-container)** — The base template
