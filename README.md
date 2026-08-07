# Agentic GraphRAG

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An **Agentic GraphRAG** question-answering system built on **GraphRAG + Agentic RAG**. It features a **hybrid routing mechanism** that adaptively selects between Standard RAG / GraphRAG / Hybrid paths based on the user's question, supports **multi-modal memory input** (text, images, tables, PDFs), and integrates **advanced RAG design patterns** (query rewriting, hybrid retrieval, re-ranking).

It can be used either as a **Python package** (`import agentic_rag`) or as a **FastAPI service** with a RESTful API.

> **🌐 Languages:** [English](README.md) · [简体中文](README.zh-CN.md)

---

## Table of Contents

- [Key Features](#key-features)
- [Web App: Browser-local PDF Chat](#web-app-browser-local-pdf-chat)
- [Environment Requirements](#environment-requirements)
- [Installation](#installation)
- [Quick Start (import and go)](#quick-start-import-and-go)
- [Multi-modal Document Ingestion](#multi-modal-document-ingestion)
- [Advanced Usage](#advanced-usage)
- [Configuration](#configuration)
- [Run the FastAPI Service](#run-the-fastapi-service)
- [REST API Reference](#rest-api-reference)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [FAQ](#faq)

---

## Key Features

| Feature | Description |
|---------|-------------|
| 🔀 **Hybrid Routing** | LLM intent classification + rule-based dual channel; auto-routes to Standard RAG / GraphRAG / Hybrid with confidence threshold and fallback |
| 🧠 **Agentic Orchestration** | Multi-step reasoning, query rewriting, answer fusion, self-reflection, conversation memory |
| 📄 **Multi-modal Memory** | Unified parsing & retrieval for text / images (OCR + LLM description) / tables / PDFs |
| 🔍 **Hybrid Retrieval** | Vector search + BM25 keyword + graph traversal, fused with RRF (Reciprocal Rank Fusion) |
| 📊 **Re-ranking** | BGE cross-encoder (local) / Cohere (API) fine-ranking |
| 🕸️ **GraphRAG** | Entity/relation extraction, knowledge graph construction, community detection, graph traversal QA |
| 🛡️ **Resilient by Design** | Graceful degradation when LLM / embedding / vector store / graph store are unavailable |

---

## Web App: Browser-local PDF Chat

A **pure front-end** PDF QA page lives in [`web/`](web/README.md). Visitors can:

1. Upload their own PDFs (parsed entirely in the browser — nothing is uploaded to any server)
2. Bring their own API Key and LLM configuration (Key stays in `localStorage`, only sent to the API endpoint they configured)
3. Build a local index (BM25 keyword search by default; optional Embedding API for vector-enhanced retrieval)
4. Ask questions against their documents, with cited sources

**Deploy it to GitHub Pages in minutes** — no backend needed:

- Repository Settings → Pages → Source: `Deploy from a branch`
- Branch: `main`, directory: `/web` → Save
- Open `https://<username>.github.io/<repo>/web/`

> If your LLM provider does not support browser CORS, self-host the optional same-origin proxy: `python -m uvicorn web.proxy:app --host 0.0.0.0 --port 8000`.

---

## Environment Requirements

- Python >= 3.11
- Virtual environment name: `agenticrag` (per project convention)
- Dependencies: `requirements.txt` + pip

---

## Installation

```bash
# Option 1: local editable install (development)
pip install -e .

# Option 2: build wheel and install (release)
pip install build
python -m build
pip install dist/agentic_rag-*.whl

# Option 3: install all optional dependencies (torch / chromadb / unstructured, etc.)
pip install agentic-rag[all]
```

### Optional dependency groups (extras)

| Group | Contents |
|-------|----------|
| `all` | Everything (equivalent to local-models + pdf + vector-db + image) |
| `local-models` | torch, torchvision, transformers, sentence-transformers |
| `pdf` | unstructured[pdf] |
| `vector-db` | chromadb, qdrant-client |
| `image` | pillow, pytesseract |
| `table` | tabula-py, camelot-py, pandas |
| `dev` | pytest, pytest-asyncio |

Core dependencies stay lightweight (fastapi, pydantic, openai, networkx, numpy, loguru, rank-bm25, jieba) so `pip install agentic-rag` works immediately in **degraded mode**; heavy dependencies live in extras.

---

## Quick Start (import and go)

```python
from agentic_rag import AgenticRAG

rag = AgenticRAG()

# Ingest a document
rag.ingest_text("人工智能公司深度智能专注于大语言模型研发。")

# Ask a question
answer = rag.query("深度智能公司是做什么的？")
print(answer)
```

### Passing configuration

```python
rag = AgenticRAG(llm_api_key="sk-...", vector_db_path="./data/vec")
```

### High-level client methods

| Method | Description |
|--------|-------------|
| `ingest(documents, ...)` | Ingest multi-modal documents (text/image/table/pdf) |
| `ingest_text(text, ...)` | Convenience: ingest a single text |
| `query(query, mode=..., top_k=..., rerank=...)` | Ask a question (auto-routing) |
| `health()` | Component connection status |
| `clear_history(conversation_id=None)` | Clear conversation history |
| `close()` | Release resources (close graph DB connection, etc.) |

---

## Multi-modal Document Ingestion

Each document item looks like:

```python
{
    "content": "...",                 # text content or base64-encoded data
    "modality": "text|image|table|pdf",
    "metadata": {"title": "...", "tags": [...]},
    "collection": "default",
}
```

The ingestion pipeline: **parse → chunk → embed → vector store → (optional) knowledge graph**.

### Multi-modal retrieval via knowledge-graph references (RAG-Anything style)

When ingesting documents (especially PDFs), the system:

1. **Records reference positions** — detects `图1 / Figure 1 / 表2 / Table 2` mentions inside each chunk (`MediaRef` with `media_id`, `label`, `page`, `offset`).
2. **Builds a media reference graph** — `chunk --references--> media` edges in the graph store (`NetworkX` / `Neo4j`), plus `media`/`chunk` nodes.
3. **Stores media assets** — extracted page images / table text are kept in `MediaRegistry` (`media_store.py`).

At query time, `enable_multimodal=True` (or config `AGR_ENABLE_MULTIMODAL_RETRIEVAL=true`) extends normal retrieval: matched chunks pull their referenced images/tables via the graph. Images are described by a **VLM** (vision-language model) before being passed to the LLM; tables are included as text.

Configure a VLM via env vars (or `POST /api/v1/config/vlm`):

```bash
AGR_VLM_MODEL=gpt-4o            # e.g. gpt-4o / qwen-vl-max / glm-4v
AGR_VLM_API_KEY=sk-...
AGR_VLM_BASE_URL=               # OpenAI-compatible base URL
```

If the VLM is not configured, multi-modal retrieval still returns image/table references (without image descriptions). The Python package exposes `AgenticRAG.save_vlm_config(...)`, and the web app shows a modal reminder when a multimodal query hits images but no VLM is configured.

**Media memory management** — extracted image base64 data is persisted to `AGR_MEDIA_STORE_PATH` (`./data/media/media_registry.json`) and only kept in memory up to `AGR_MEDIA_MAX_MEMORY_MB` (default 512 MB). When the limit is exceeded, older image payloads are unloaded from RAM and lazily reloaded from disk on demand, so very large PDFs won't exhaust memory.

---

## Advanced Usage

### Direct orchestrator access

```python
from agentic_rag import AgenticOrchestrator, QueryRequest

result = orchestrator.query(
    QueryRequest(
        query="2024年人工智能领域有哪些重大突破？",
        mode="auto",          # auto | standard | graph | hybrid
        top_k=5,
        rerank=True,
    )
)
```

### Query rewriting & hybrid retrieval

The system automatically applies query rewriting (expansion / decomposition / HyDE), hybrid retrieval (vector + BM25 + graph), RRF fusion, and re-ranking — no extra configuration needed. External conversation history passed via `QueryRequest.history` is honored, falling back to internal session memory when absent.

---

## Configuration

All settings are managed by `pydantic-settings` and can be overridden via environment variables with the `AGR_` prefix (or a `.env` file). See [`.env.example`](.env.example) for all options.

```python
# config.py (key settings)
class Settings:
    # Vector DB
    vector_db_type: str = "chroma"        # chroma | qdrant
    vector_db_path: str = "./data/vector_db"

    # Graph DB
    graph_db_type: str = "networkx"       # networkx | neo4j

    # Embedding model
    embedding_model: str = "BAAI/bge-m3"

    # Re-ranker model
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # LLM
    llm_model: str = "gpt-4o"
    llm_api_key: str = ""
    llm_base_url: str = ""

    # Retrieval params
    top_k_initial: int = 20
    top_k_rerank: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 128
```

---

## Run the FastAPI Service

```bash
# Using the project virtual environment
agenticrag\Scripts\python.exe -m agentic_rag.main
# or
uvicorn agentic_rag.main:app --host 0.0.0.0 --port 8000
```

Interactive API docs: `http://localhost:8000/docs`

---

## REST API Reference

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/query` | Question answering |
| POST | `/api/v1/ingest` | Document ingestion |
| POST | `/api/v1/feedback` | Feedback submission |
| GET  | `/api/v1/health` | Health check |
| GET  | `/api/v1/collections` | List collections |
| DELETE | `/api/v1/collections/{name}` | Delete a collection |

### Query

```json
// Request
{
  "query": "2024年人工智能领域有哪些重大突破？",
  "conversation_id": "conv_001",
  "mode": "auto",
  "top_k": 5,
  "rerank": true
}

// Response
{
  "answer": "...",
  "route": "hybrid",
  "confidence": 0.95,
  "sources": [{"doc_id": "doc_001", "content": "...", "score": 0.92, "modality": "text"}],
  "conversation_id": "conv_001",
  "latency_ms": 1250
}
```

Optional API Key auth: set `AGR_API_KEY`; requests must then carry `X-API-Key`.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        API Layer (FastAPI)                       │
│  POST /query  POST /ingest  POST /feedback  GET /health          │
├──────────────────────────────────────────────────────────────────┤
│                       Agentic Orchestrator                       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐   │
│  │ Intent   │ │ Query    │ │ Path     │ │ Answer Synthesis │   │
│  │ Classify │ │ Rewrite  │ │ Select   │ │ & Verification   │   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘   │
├──────────┬──────────────────┬──────────────────┬────────────────┤
│ Standard │   GraphRAG       │   Multi-Modal    │   Memory       │
│ RAG      │   Engine         │   Parser         │   Store        │
├──────────┴──────────────────┴──────────────────┴────────────────┤
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │ Vector   │ │ Graph    │ │ Keyword  │ │ Hybrid   │          │
│  │ Search   │ │ Traverse │ │ (BM25)   │ │ Retrieve │          │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘          │
├──────────────────────────────────────────────────────────────────┤
│  Vector DB (ChromaDB)    │  Graph DB (NetworkX/Neo4j)           │
│  Embedding Models        │  LLM (OpenAI/LiteLLM)                │
│  Reranker Models         │  Document Parsers                    │
└──────────────────────────────────────────────────────────────────┘
```

### Query processing flow

```
User query → Intent classification → Query rewriting → Hybrid retrieval → Re-ranking → Answer generation
    ↓              ↓                    ↓                  ↓                 ↓                ↓
  History      Route decision       Rewritten        Multi-source      Re-ranked       Final answer
                                    variants           fusion           results
```

---

## Project Structure

```
F:\intern\Agent\AgenticRag\
├── SPEC.md                    # Specification
├── requirements.txt           # Dependencies
├── pyproject.toml             # Build config (setuptools + extras)
├── README.md                  # This file (English)
├── README.zh-CN.md            # Chinese README (alternative)
├── agentic_rag/
│   ├── __init__.py
│   ├── main.py                # App entry (FastAPI)
│   ├── config.py              # Configuration management
│   ├── api/
│   │   ├── routes.py          # API routes
│   │   └── models.py          # Request/response models
│   ├── core/
│   │   ├── orchestrator.py    # Agentic orchestrator
│   │   ├── hybrid_router.py   # Hybrid router
│   │   └── query_rewriter.py  # Query rewriter
│   ├── rag/
│   │   ├── standard_rag.py    # Standard RAG engine
│   │   ├── graph_rag.py       # GraphRAG engine
│   │   └── hybrid_retriever.py# Hybrid retriever
│   ├── memory/
│   │   ├── multi_modal_parser.py  # Multi-modal parser (text / image / table / PDF + media refs)
│   │   ├── media_store.py         # Media asset registry (image base64 / table text)
│   │   ├── vector_store.py    # Vector store
│   │   └── graph_store.py     # Graph store (+ media reference graph)
│   ├── processing/
│   │   ├── embedders.py       # Embedders
│   │   ├── reranker.py        # Re-ranker
│   │   └── chunker.py         # Document chunker
│   └── utils/
│       └── helpers.py         # Helpers
├── web/
│   ├── index.html             # Browser-local PDF chat page
│   ├── style.css
│   ├── app.js
│   ├── proxy.py               # Optional same-origin proxy (FastAPI)
│   └── README.md
└── tests/
    ├── test_router.py
    ├── test_retriever.py
    └── test_api.py
```

---

## Testing

```bash
# Using the project virtual environment
agenticrag\Scripts\python.exe -m pytest tests -q
```

All tests pass (14 tests covering router, retriever, chunking, BM25, and API endpoints).

---

## FAQ

**Q1: `import agentic_rag` fails with ModuleNotFoundError?**
Make sure you activated the `agenticrag` virtual environment and installed the package (`pip install -e .` or the built wheel).

**Q2: Can I use it without an LLM API key?**
Yes. The system degrades gracefully: rule-based routing (keyword classification) + fallback answer generation + rule-based entity extraction. Configure `AGR_LLM_API_KEY` to unlock full capabilities.

**Q3: BGE model download fails?**
Local embedding/re-ranking models are downloaded from HuggingFace. Point `AGR_EMBEDDING_MODEL` to a locally downloaded model path, or switch to OpenAI embeddings (`AGR_EMBEDDING_MODEL=text-embedding-3-small` + API key).

**Q4: How do I switch to Neo4j?**
Set `AGR_GRAPH_DB_TYPE=neo4j` plus `AGR_NEO4J_URI/USER/PASSWORD`.

**Q5: How do I rebuild the wheel?**
```bash
agenticrag\Scripts\python.exe -m build --wheel --no-isolation
```

---

## Roadmap

- [x] Core architecture
- [x] Hybrid routing
- [x] Multi-modal parsing
- [x] Multi-modal retrieval via knowledge-graph media references (RAG-Anything style) + VLM
- [x] Advanced retrieval (query rewriting / hybrid retrieval / re-ranking)
- [x] RESTful API
- [x] Packaging (wheel / import and go)
- [ ] Streaming output (SSE)
- [ ] Enhanced multi-turn conversation memory
- [ ] Online learning from user feedback
- [ ] Graph visualization UI
- [ ] Distributed deployment

---

## License

MIT
