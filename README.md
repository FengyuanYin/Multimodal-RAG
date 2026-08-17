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

Interactive startup displays a six-line purple-to-cyan `AutoMemory` brand in ANSI-capable terminals. Narrow terminals use a compact title. `--no-color`, `NO_COLOR`, redirected output, `-p`, and pipe mode never emit colored startup output.

### Guided API setup

Run `/setup` inside AutoMemory for a step-by-step cloud configuration wizard:

```text
AutoMemory> /setup
Configure LLM chat?
  1. Configure or replace
  2. Keep current settings
```

The wizard supports OpenAI, DeepSeek, SiliconFlow, and custom OpenAI-compatible endpoints for chat, embeddings, and vision; SiliconFlow or Cohere-compatible reranking; official or self-hosted MinerU; and DuckDuckGo or Tavily search. Provider defaults remain editable. When SiliconFlow reranking is selected, AutoMemory can reuse a SiliconFlow key already entered for chat, embeddings, or vision after confirmation. Some providers do not offer every model capability, so presets marked “model required” require a model supported by that account or compatible gateway.

Values remain in memory until the final confirmation. `back`, `skip`, and `cancel` are available during setup. After saving, AutoMemory can issue a real, minimal connection test and distinguish authentication, quota/rate-limit, network, model, and malformed-response failures. A failed test does not delete the saved configuration.

### Recommended workflow and fixed RAG modes

```text
/setup
/mode balanced
/kb create "My research"
/add "D:\Documents\paper.pdf"
/docs
/s What method does the paper propose?
```

`/mode` exposes four product presets; their internal chunk, candidate, fusion, and window parameters are fixed so users do not need to assemble an algorithm manually.

| Mode | Retrieval pipeline |
|---|---|
| `fast` | BM25 only; fastest and requires no embedding API. |
| `balanced` | BM25 + cloud embedding; the default for new installations. |
| `multimodal` | Balanced plus figures, tables, captions, and the document/page/media reference graph. |
| `advanced` | Query rewriting, BM25, embedding, multimodal evidence, entity GraphRAG, reference graph, stable RRF fusion, cloud reranking, post-rerank VLM image routing, then adjacent regular-chunk context expansion. It does not build an extra small-chunk index. |

### Cost-aware Advanced VLM routing

Automatic image understanding is exclusive to `advanced`. AutoMemory first completes retrieval, fusion, cloud reranking, and final top-k selection. It then takes every unique image referenced by those final chunks—there is no image-count cap—and sends only uncached images to the configured VLM. Images attached only to candidates eliminated by reranking never incur a VLM call.

One primary call classifies and reconstructs each image: data charts become Markdown tables, structure diagrams become Mermaid text, and other images become factual visual descriptions. A second general-vision call is made only when chart-table or Mermaid validation fails. Successful results are cached by media checksum, VLM profile, and prompt version, so later queries and process restarts can reuse them without another call. Individual image failures are recorded in `trace.advanced_vlm` and do not discard the selected text evidence.

Use `/kb` to list knowledge bases. `/kb create <name>` creates and selects one; `/kb use <id-or-name>` switches; `/kb rename ...` renames; `/kb delete ...` deletes after confirmation. `/add`, `/docs`, `/remove`, `/s`, and `/graph` always use the current knowledge base. Existing categories are retained as knowledge bases during upgrade.

Advanced graph data can be exported without a GUI:

```text
/graph                         # combined graph
/graph entity entities.png
/graph reference references.png
```

Each export writes a Matplotlib PNG and a same-name JSON provenance file under the AutoMemory exports directory. Large graphs produce a deterministic readable subgraph and report the original/exported node and edge counts.

### Chat and commands

Commands use Windows-friendly tokenization. Wrap paths or names containing spaces in double quotes. An ID may be shortened to a unique prefix. Options in `[brackets]` are optional; `--force` and `--yes` skip interactive confirmation.

#### Chat input

| Input | Meaning | Output |
|---|---|---|
| `<message>` | Direct cloud-LLM chat. The knowledge base is not searched. | A streamed answer; the user and assistant messages are saved in the active conversation. |
| `/s <question>` | Search only the current knowledge base with the selected fixed mode, then ask the cloud LLM with retrieved context. `/s` must be the exact lowercase first token. | A streamed grounded answer, numbered sources, and a retrieval trace available through `/trace`. |

#### Core and diagnostics

| Command and input | Meaning | Output / side effect |
|---|---|---|
| `/help`, `/help all`, or `/help <command>` | Show the ten primary commands, all compatibility commands, or one command's usage. | A concise default view or the complete advanced command list. |
| `/version` | Show the running AutoMemory version. | `AutoMemory 0.4.0`. |
| `/diagnose [--errors]` | Check local databases, data directory, cloud credential presence, and required CLI dependencies. `--errors` also includes recent redacted errors from this process. | One `OK`, `DEGRADED`, or `ERROR` line per check; no secret values. It does not make provider API calls. |
| `/path` | Show the isolated data locations used by this process. | Home, exports, and logs paths. |
| `/exit` or `/quit` | Close the REPL after the current command returns. | Clean process exit; databases and clients are closed. |

#### Conversations and long-term memory

| Command and input | Meaning | Output / side effect |
|---|---|---|
| `/new [title]` | Create and select a conversation; default title is `New conversation`. | New conversation ID and title. |
| `/sessions` | List saved conversations. | IDs and titles; `*` marks the active conversation. |
| `/use <conversation-id>` | Switch to a conversation using its full ID or unique prefix. | Selected conversation ID. |
| `/rename <title>` | Rename the active conversation. | Confirmation containing the new title. |
| `/clear [--force]` | Remove every message from the active conversation. | Confirmation prompt unless forced, then `Current conversation cleared`. The conversation itself remains. |
| `/delete [conversation-id] [--force]` | Delete the selected conversation, or the active one when no ID is supplied. | Confirmation prompt unless forced, deletion result, and automatic selection/creation of another active conversation. |
| `/memory` or `/memory list` | List long-term memory entries. | Entry status (`on`/`off`), ID, and content. |
| `/memory add <content>` | Add an enabled long-term memory for future chat context. | New memory ID. |
| `/memory enable\|disable <memory-id>` | Include or exclude a saved memory without deleting it. | Updated memory ID and state. |
| `/memory delete <memory-id>` | Permanently remove a memory. | Deleted memory ID. |

#### Knowledge base and retrieval

| Command and input | Meaning | Output / side effect |
|---|---|---|
| `/mode [fast\|balanced\|multimodal\|advanced]` | List or select one fixed RAG preset. | Persisted current mode and any derived-index preparation progress. |
| `/kb [list\|create\|use\|rename\|delete] ...` | Manage and select knowledge bases. Creating a base selects it automatically. | Knowledge-base IDs, names, document counts, and current marker. |
| `/add <path> [path...] [--vlm]` | Import one or more local PDF, text/Markdown, image, or spreadsheet files into the current knowledge base. | Parse/chunk/index progress and imported-or-duplicate result per file. |
| `/docs` | List documents in the current knowledge base. | Document ID, title, parser, status, and chunk/media counts. |
| `/doc <document-id>` | Inspect one document using a full ID or unique prefix. | Title, source, parser, status, pages, chunks, and media counts. |
| `/remove <document-id> [--force]` | Delete a current-base document, its chunks, embeddings, dual-graph records, and stored media. | Confirmation prompt unless forced, followed by the deleted document ID. |
| `/graph [entity\|reference\|combined] [filename.png]` | Export the current knowledge base graph with Matplotlib Agg. | PNG, same-name provenance JSON, and graph size statistics. |
| `/category` or `/category list` | Legacy compatibility alias for the underlying knowledge-base categories; prefer `/kb`. | Category IDs and names. |
| `/category add <name>` | Create a category. | New category ID and name. |
| `/category rename <id> <name>` | Rename a category. | Renamed category ID. |
| `/category delete <id> [--force]` | Delete an empty category. Documents must be removed or moved first. | Confirmation prompt unless forced, then deleted category ID. |
| `/reindex [--force]` | Rebuild keyword and Milvus vector indexes from stored chunks; `--force` recreates only AutoMemory-managed Milvus collections. | Keyword count plus vector-ready/degraded counts. It does not reparse source files. |
| `/trace` | Show details of the most recent `/s` retrieval in the current process. | JSON containing retrieval mode, channels, scores, fallbacks, and scope; otherwise a “no RAG query” message. |
| `/export <media-id> [filename]` | Copy a stored media asset to AutoMemory's exports directory. | Absolute exported file path. The optional filename is sanitized. |
| `/mineru <pdf-path> [--category id] [--selfhost]` | Parse a PDF through the configured MinerU service and import it into the current knowledge base by default; `--category` overrides the target and `--selfhost` overrides the saved parser mode for this command. | Upload/poll/download/ingestion progress, imported document summary, task status, and rebuilt index. |
| `/context open <document-id>` | Open one MinerU document as an isolated full-Markdown workspace. Normal messages then use the complete immutable Markdown with the main LLM; `/s` remains knowledge retrieval. | Workspace ID, Markdown source/size, document and main model. |
| `/context status\|leave\|clear` | Inspect, leave, or clear the active full-document workspace. | Current document state or lifecycle result; clearing preserves the source Markdown. |
| `/context files\|read\|export\|delete-file` | List, read, export, or confirm deletion of managed Markdown produced by long answers, image analysis, summaries, or model notes. | Stable file IDs and bounded file content/export/deletion result. |

#### Web and evaluation

| Command and input | Meaning | Output / side effect |
|---|---|---|
| `/search <keywords>` | Search with the configured DuckDuckGo or Tavily provider. | Numbered results containing title, URL, and snippet; the numbered list is retained for the next `/fetch`. |
| `/fetch <result-number\|url> [--category id] [--yes]` | Fetch a public page by URL or by number from the last search, preview its readable text, then optionally import it into the current knowledge base by default; `--category` overrides the target. | Title, final URL, character count, and preview; after confirmation, an import result and rebuilt index. `--yes` imports immediately. |
| `/eval <dataset.json> [--mode keyword\|vector\|hybrid\|multimodal] [--top-k N] [--scope id] [--export]` | Run deterministic retrieval evaluation. Each JSON case requires `query`; `expected` and `expected_media` enable relevance metrics. | Progress plus summary JSON for Precision@K, Recall@K, MRR, nDCG@K, media recall, and latency. `--export` also writes the full run under exports. |

#### API configuration and credentials

| Command and input | Meaning | Output / side effect |
|---|---|---|
| `/setup` | Guided configuration for LLM, embedding, VLM, reranker, MinerU, and Web search. Settings remain staged until final confirmation. | Redacted summary, secure credential saves, config reload, and optional real connection tests. `back`, `skip`, and `cancel` are supported. |
| `/config` or `/config list` | Show every non-secret setting. | Formatted JSON; credentials are never included. |
| `/config get <key>` | Read a dotted config key, for example `llm.model` or `retrieval_mode`. | JSON value. |
| `/config set <key> <value>` | Validate and save a non-secret setting. JSON literals such as `5`, `true`, and quoted strings are accepted. | Saved-key confirmation; services and derived indexes are reloaded. |
| `/config unset <key>` | Reset one key to its AutoMemory default. | Reset confirmation; services are reloaded. |
| `/config test [llm\|embedding\|vlm\|reranker\|mineru\|web\|all]` | Send a real, minimal request to one service or all services. Default is `all`. | Per-service latency and stable status (`success`, auth, rate/quota, network, model, response, or not configured), followed by summary JSON. API usage may incur a small provider charge. |
| `/secret` or `/secret status` | Show where each supported credential comes from. | `environment`, Windows Credential Manager/session, or `not-configured`; never the value. |
| `/secret set <name>` | Securely enter and store one credential. Do not append the key to the command. | Hidden prompt, credential-source confirmation, and service reload. |
| `/secret delete <name>` | Delete a stored credential. Environment overrides are not modified. | Deleted/not-stored result and service reload. |
| `/secret test <name>` | Map a credential to its service and perform the same real probe as `/config test`. | Probe line plus credential status/code. |

Credential names are `llm_api_key`, `embedding_api_key`, `vlm_api_key`, `reranker_api_key`, `mineru_api_key`, `tavily_api_key`, and the optional `milvus_token`.

### Milvus vector storage

Both the CLI and API/SDK use Milvus. The default connection is `http://localhost:19530`, database `default`, with no authentication. API/SDK settings use the `AGR_MILVUS_*` environment variables shown in `.env.example`. AutoMemory CLI stores the non-secret `milvus_uri`, `milvus_database`, `milvus_collection`, and `milvus_timeout_seconds` fields in its config JSON; an optional token must be supplied through `AUTOMEMORY_MILVUS_TOKEN`/`AGR_MILVUS_TOKEN` or `/secret set milvus_token`.

Physical collection names include a schema version and vector dimension (for example, `automemory_vectors_v1_d1536`). Existing SQLite/legacy-backend vectors are not copied. Run `/reindex` to generate missing Milvus vectors from stored chunks, or `/reindex --force` to recreate only collections managed by the configured AutoMemory prefix. If Milvus cannot be reached, vector search is marked degraded while BM25 and graph channels remain available.

### Full-document Markdown workspace

MinerU imports now preserve the complete Markdown beside chunks and media. Run `/doc <id>` to check availability, then `/context open <id>`. One workspace binds one document and has history isolated from direct chat, `/s`, and long-term memory. The complete Markdown is never truncated or replaced by a summary. The main LLM always performs the final reasoning; it may request one associated image at a time, in which case only that image is sent to the configured VLM and the structured result is returned to the main LLM.

Defaults target a 1M-token model: 920k maximum input, compaction at 850k toward 780k, 32k output reserve and 48k safety reserve. Only mutable workspace history is summarized, with explicit summary tags and the latest six turns retained. Answers estimated above 12k tokens are stored as managed Markdown; later prompts contain a clearly incomplete 1.5k-token head/tail preview and stable file ID. AutoMemory keeps the system/document/complete-Markdown prefix deterministic to encourage provider prompt caching, but reports a cache hit only when the provider returns cached-token usage. Full-document calls can be expensive.

Model file tools accept managed IDs only. They cannot traverse arbitrary paths, execute files, overwrite source Markdown/PDF/media, or modify configuration and databases.

#### Process arguments and output behavior

| Startup input | Meaning / output |
|---|---|
| `AutoMemory.exe -p "<message or command>"` | Run once and exit. Answers/results go to stdout; errors go to stderr. Use `-p "/s question"` for retrieval. Interactive-only commands such as `/setup` are rejected. |
| `AutoMemory.exe --home <absolute-path>` | Use a different isolated data directory for config, databases, media, logs, cache, and exports. |
| `AutoMemory.exe --no-color` | Disable ANSI color. |
| `AutoMemory.exe --plain` | Use plain line-oriented stdin mode: no banner, prompt, completion, or ANSI color. Each entered line is executed; send EOF to finish. |
| `AutoMemory.exe --debug` | Include exception type/details for unexpected internal errors. Secrets remain redacted from recorded diagnostics. |
| `AutoMemory.exe --version` | Print the version and exit. |
| Piped lines, for example `Get-Content commands.txt \| AutoMemory.exe` | Execute one input per line without banner/color; stop on `/exit` or the first error. |

`Ctrl+C` cancels active work. At an idle prompt, press it twice within 1.5 seconds to exit. EOF (`Ctrl+Z`, then Enter, on Windows) also exits.

### Secure cloud credentials

The `/setup` wizard collects keys through a hidden prompt that is excluded from terminal history. On Windows keys are stored by Windows Credential Manager, never in `config.json`, SQLite, logs, or exports. Environment variables override stored credentials:

```bash
# OpenAI or another OpenAI-compatible LLM endpoint
export AUTOMEMORY_LLM_API_KEY="..."

# Official MinerU and optional Tavily search
export AUTOMEMORY_MINERU_API_KEY="..."
export AUTOMEMORY_TAVILY_API_KEY="..."
```

In PowerShell, use `$env:AUTOMEMORY_LLM_API_KEY="..."` for the current terminal session. Advanced users can continue to use `/config get|set|unset|list`, `/secret status|set|delete`, `/config test [service|all]`, and `/secret test <credential>`. Tests now perform real bounded provider requests rather than checking only whether a key exists. Embeddings, image understanding, and reranking use cloud API profiles; keyword retrieval remains available without an embedding API.

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

# Option 3: install all optional dependencies (torch / pymilvus / unstructured, etc.)
pip install agentic-rag[all]
```

### Optional dependency groups (extras)

| Group | Contents |
|-------|----------|
| `all` | Everything (equivalent to local-models + pdf + vector-db + image) |
| `local-models` | torch, torchvision, transformers, sentence-transformers |
| `pdf` | unstructured[pdf] |
| `vector-db` | pymilvus |
| `image` | pillow, pytesseract |
| `table` | tabula-py, camelot-py, pandas |
| `dev` | pytest, pytest-asyncio |

Core dependencies include `pymilvus` because Milvus is the unified vector backend. If Milvus is unavailable, keyword and graph retrieval remain available in degraded mode.

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
rag = AgenticRAG(llm_api_key="sk-...", milvus_uri="http://localhost:19530")
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
    vector_db_type: str = "milvus"
    milvus_uri: str = "http://localhost:19530"
    milvus_database: str = "default"
    milvus_collection: str = "agentic_rag_vectors"
    milvus_token: str | None = None

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
│  Vector DB (Milvus)      │  Graph DB (NetworkX/Neo4j)           │
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
