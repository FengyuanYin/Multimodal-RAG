# Agentic GraphRAG

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An **Agentic GraphRAG** question-answering system built on **GraphRAG + Agentic RAG**. It features a **hybrid routing mechanism** that adaptively selects between Standard RAG / GraphRAG / Hybrid paths based on the user's question, supports **multi-modal memory input** (text, images, tables, PDFs), and integrates **advanced RAG design patterns** (query rewriting, hybrid retrieval, re-ranking).

It can be used either as a **Python package** (`import agentic_rag`) or as a **FastAPI service** with a RESTful API.

## AutoMemory terminal CLI and Windows EXE

AutoMemory is an additional, line-oriented terminal application inspired by coding assistants such as Claude Code. It does not replace or change the existing Web app, REST API, or Python package. It uses cloud APIs for every AI model; there is no local model runtime.

### Install and run

```bash
git clone https://github.com/FengyuanYin/Multimodal-RAG.git
cd Multimodal-RAG
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

pip install -e ".[cli]"
automemory
```

You can also run `python -m agentic_rag.cli`. For Windows, download the `AutoMemory-windows-x64` artifact from the latest **Build AutoMemory Windows EXE** GitHub Actions run and start `AutoMemory.exe` in PowerShell or Command Prompt. The EXE is a single console program; no Python installation is required.

To build the EXE locally with Python 3.11, run `powershell -ExecutionPolicy Bypass -File .\packaging\build_windows.ps1`. The result is `dist\AutoMemory.exe`.

Set `AUTOMEMORY_HOME` to an absolute path or pass `--home <absolute-path>` to override `%APPDATA%\AutoMemory`. AutoMemory keeps its state, knowledge database, media, exports, cache, and logs in this isolated directory; it does not reuse or mutate the Web app's browser storage.

### Chat and commands

- Type a normal message to call the configured cloud LLM directly, without searching the knowledge base.
- Only a message beginning with the exact `/s` prefix enables local knowledge retrieval: `/s What does the paper conclude?`
- `/search <keywords>` searches the Web; `/fetch <url>` captures a public page; `/mineru <pdf>` uses the cloud MinerU API.
- `/add <path>`, `/docs`, `/sessions`, `/memory`, `/eval`, `/config`, `/secret`, `/diagnose`, and `/help` provide the remaining workflows.
- `Ctrl+C` cancels active work. Use `/exit` or EOF to close the program.
- For automation, `AutoMemory.exe -p "question"` writes the streamed answer to stdout and errors to stderr.

### Secure cloud credentials

Run `/secret set llm_api_key` and enter the key at the hidden prompt. On Windows it is stored by Windows Credential Manager, never in `config.json`, SQLite, logs, or exports. Environment variables override stored credentials:

```bash
# OpenAI or another OpenAI-compatible LLM endpoint
export AUTOMEMORY_LLM_API_KEY="..."

# Official MinerU and optional Tavily search
export AUTOMEMORY_MINERU_API_KEY="..."
export AUTOMEMORY_TAVILY_API_KEY="..."
```

In PowerShell, use `$env:AUTOMEMORY_LLM_API_KEY="..."` for the current terminal session. Configure model names and credential-free Base URLs with `/config`. Embeddings, image understanding, and reranking also use cloud API profiles. Keyword retrieval remains available without an embedding API.

### Knowledge sources and evaluation

The CLI imports local PDF, text/Markdown, image, and table files; captures readable Web pages; and parses PDFs through the official or a self-hosted MinerU service. DuckDuckGo search requires no key, while Tavily requires `AUTOMEMORY_TAVILY_API_KEY`. Requests run from the local EXE and are not subject to GitHub Pages browser CORS restrictions. Public Web capture blocks private/reserved network targets; a self-hosted MinerU URL is allowed only when explicitly selected.

Evaluation datasets are JSON arrays (or an object with `cases` / `items`) containing at least a `query`. Optional `expected` document IDs and `expected_media` IDs enable Precision@K, Recall@K, MRR, nDCG@K, and media recall. Results are atomically exported inside AutoMemory's exports directory.

```json
{
  "cases": [
    {"id": "q1", "query": "What is the main conclusion?", "expected": ["doc_123"], "expected_media": ["figure1"]}
  ]
}
```

> **🌐 Languages:** [English](README.md) · [简体中文](README.zh-CN.md)

---

## Table of Contents

- [Key Features](#key-features)
- [Web App: Browser-local PDF Chat](#web-app-browser-local-pdf-chat)
- [GitHub Pages Proxy Setup](#github-pages-proxy-setup-web-search-and-mineru)
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

The `web/` folder is a **browser-local** PDF Q&A tool (BYOK — Bring Your Own Key): upload PDFs, or **search the web / paste URLs to grab web pages into your knowledge base**, build a local index (BM25 + optional embeddings), and chat with citations. Documents stay in the browser unless you explicitly choose the MinerU parser; API keys are kept in browser session storage and are sent only to the endpoints you configure.

See [web/README.md](web/README.md) for details.

A static browser-local PDF QA page lives in [`web/`](web/README.md). Visitors can:

1. Upload their own PDFs (parsed entirely in the browser — nothing is uploaded to any server)
2. **Search the web / paste URLs to grab web pages into the knowledge base** (via the optional service proxy; DuckDuckGo free search or Tavily)
3. Bring their own API Key and LLM configuration (secrets stay in `sessionStorage` and are sent only to the configured endpoint)
4. Build a local index (BM25 keyword search by default; optional Embedding API for vector-enhanced retrieval)
5. Ask questions against their documents, with cited sources

### Chat routing: direct by default, `/s` for knowledge search

The chat input uses explicit per-message routing:

- Enter a normal message to chat directly with the configured LLM. The document knowledge base is not searched and no document citations are added.
- Prefix a message with the exact, lowercase `/s` command to search the selected knowledge-base scope. The command is removed before retrieval and generation, while the original message remains visible in conversation history.

```text
Explain retrieval-augmented generation.
# Direct LLM chat; uploaded documents are not searched.

/s Summarize the main conclusions of this paper.
# Searches the knowledge base, then answers with document citations.
```

`/s` is case-sensitive and must be a complete first token after optional leading whitespace. `/search topic`, `/S topic`, and `/sTopic` are ordinary direct-chat messages. Sending `/s` without a question displays a validation prompt and makes no LLM or retrieval request. Conversation history and enabled user memory remain available in both routes; keyword, vector, hybrid, and multimodal retrieval settings apply only to `/s` requests.

**Deploy the static interface to GitHub Pages in minutes:**

- Repository Settings → Pages → Source: `Deploy from a branch`
- Branch: `main`, directory: `/web` → Save
- Open `https://<username>.github.io/<repo>/web/`

Local PDF parsing and BM25 retrieval need no backend. Web search, page fetching, MinerU, and LLM providers without browser CORS require the proxy described below.

---

## GitHub Pages Proxy Setup (Web Search and MinerU)

GitHub Pages serves static files only; it cannot run [`web/proxy.py`](web/proxy.py). To use Web Search or the official MinerU API from the hosted interface, deploy the proxy as a separate public HTTPS service and enter that service URL in the web app.

```text
GitHub Pages UI
      │ HTTPS
      ▼
web/proxy.py (Render or another Python host)
      ├── DuckDuckGo / Tavily search and page fetching
      └── MinerU official API
```

### Option A: Run the proxy locally

From the repository root:

```bash
pip install -r requirements.txt
python -m uvicorn web.proxy:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/proxy/health`. A healthy proxy returns:

```json
{"status":"ok"}
```

In the GitHub Pages app, open **Knowledge Base → Web Search & Fetch**, enter `http://127.0.0.1:8000` as the **Service Proxy URL**, then select **Save and Test**. This local address works only for the user running the proxy on the same computer.

### Option B: Deploy the proxy to Render

1. Sign in to [Render](https://render.com/) and create a **Web Service** from this GitHub repository.
2. Select the `main` branch and the Python runtime.
3. Use the following commands:

   ```text
   Build Command: pip install -r requirements.txt
   Start Command: uvicorn web.proxy:app --host 0.0.0.0 --port $PORT
   ```

4. Add this environment variable:

   ```text
   AGR_PROXY_ALLOWED_ORIGINS=https://fengyuanyin.github.io
   ```

5. Deploy, then verify `https://<your-service>.onrender.com/proxy/health`.
6. In the web app, set **Service Proxy URL** to `https://<your-service>.onrender.com`. Do not append `/proxy` or an endpoint path.

The same commands work on other Python hosting platforms. Production deployments must use HTTPS.

### Enable Web Search and page fetching

After the proxy health check succeeds:

- **DuckDuckGo** requires no API key.
- **Tavily** requires a Tavily API key entered in the Knowledge Base panel.
- Pasted URLs and selected search results are fetched through `/proxy/web/fetch` and added to the browser-local knowledge base.

### Enable MinerU

1. Open **Settings** and select **MinerU → Official API** as the parser.
2. Enter a valid MinerU API key.
3. Leave **MinerU Proxy URL** empty to reuse the shared Service Proxy URL, or enter a separate trusted proxy URL.
4. Save the settings and upload the PDF again.

The browser sends the PDF and MinerU key to your configured proxy. The proxy creates an official MinerU batch job, uploads the PDF, polls the job, downloads the result archive, and returns normalized pages and media to the browser. The proxy does not persist API keys or uploaded files.

### Security and troubleshooting

- Never commit API keys. Revoke any key that has appeared in a screenshot, chat, issue, or commit.
- Only use a proxy you control and trust. A publicly reachable proxy should be protected with authentication, rate limiting, request-size limits, HTTPS, and monitoring before production use.
- `Failed to fetch` usually means the proxy URL is wrong, the service is asleep or unavailable, HTTPS/CORS is misconfigured, or a browser private-network request was denied.
- A MinerU stage-specific error now identifies whether task creation, PDF upload, status polling, result download, or archive parsing failed. Retry transient timeout errors; verify the MinerU key and account quota for repeated authorization or quota failures.
- Some free hosting plans sleep when idle. The first request after wake-up may take longer than usual, and complex MinerU documents can take several minutes.

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
