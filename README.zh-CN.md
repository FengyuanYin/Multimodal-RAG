# Agentic GraphRAG 系统

## AutoMemory 终端 CLI 与 Windows EXE

AutoMemory 是额外提供的单终端、逐行交互程序，使用体验接近 Claude Code。它不替换、不修改现有 Web 页面、REST API 或 Python 包。所有 AI 模型均通过云 API 调用，不运行本地模型。

### 安装与启动

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

也可以执行 `python -m agentic_rag.cli`。Windows 用户可在 GitHub Actions 的 **Build AutoMemory Windows EXE** 最新运行记录中下载 `AutoMemory-windows-x64` 构建产物，解压后直接在 PowerShell 或 CMD 中运行 `AutoMemory.exe`，无需安装 Python。

若要在本机使用 Python 3.11 构建，执行 `powershell -ExecutionPolicy Bypass -File .\packaging\build_windows.ps1`，输出文件为 `dist\AutoMemory.exe`。

通过绝对路径环境变量 `AUTOMEMORY_HOME` 或 `automemory --home <绝对路径>` 指定数据目录；默认目录为 `%APPDATA%\AutoMemory`。会话、记忆、知识库、媒体、导出、缓存和日志均存放在这个隔离目录，不会复用或修改 Web 页面的浏览器数据。

交互启动时，支持 ANSI 的终端会显示六行紫色到青色渐变的 `AutoMemory` 艺术字；窄终端自动使用紧凑标题。`--no-color`、`NO_COLOR`、输出重定向、`-p` 和管道模式不会输出彩色启动内容。

### API 配置向导

进入 AutoMemory 后执行 `/setup`，即可逐步配置云 API：

```text
AutoMemory> /setup
Configure LLM chat?
  1. Configure or replace
  2. Keep current settings
```

向导支持 OpenAI、DeepSeek、SiliconFlow 和自定义 OpenAI-compatible 的对话、嵌入与视觉模型；SiliconFlow 或 Cohere-compatible 重排序；MinerU 官方或自托管服务；DuckDuckGo 或 Tavily 搜索。预设值均可修改。选择 SiliconFlow 重排序时，经确认可复用本轮已为对话、嵌入或视觉模型填写的 SiliconFlow Key。部分服务商并不原生提供所有模型能力，标有“model required”的选项需要填写该账号或兼容网关实际支持的模型。

最终确认前，所有值只保存在内存中；向导支持 `back`、`skip` 和 `cancel`。保存后可执行真实、低成本连接测试，并区分认证、余额/限流、网络、模型和响应格式错误。测试失败不会删除已经确认的配置。

### 对话与命令

- 普通输入默认直接调用云端 LLM，不检索知识库。
- 只有消息以精确的 `/s` 前缀开头才检索知识库，例如：`/s 这篇论文的结论是什么？`
- `/search <关键词>` 搜索网页，`/fetch <网址>` 抓取网页，`/mineru <PDF路径>` 调用云端 MinerU。
- `/add`、`/docs`、`/sessions`、`/memory`、`/eval`、`/config`、`/secret`、`/diagnose` 和 `/help` 提供其余管理功能。
- `Ctrl+C` 取消当前任务，`/exit` 或 EOF 退出。
- 自动化调用可使用 `AutoMemory.exe -p "问题"`；回答写入 stdout，错误写入 stderr。

### 密钥安全

`/setup` 通过不进入终端历史的隐藏提示采集密钥。Windows 下密钥保存到 Windows 凭据管理器，不写入 `config.json`、SQLite、日志或导出文件。环境变量优先级高于已保存凭据：

```powershell
$env:AUTOMEMORY_LLM_API_KEY="..."
$env:AUTOMEMORY_MINERU_API_KEY="..."
$env:AUTOMEMORY_TAVILY_API_KEY="..."
automemory
```

高级用户仍可使用 `/config get|set|unset|list`、`/secret status|set|delete`、`/config test [service|all]` 和 `/secret test <credential>`。测试命令现在会执行真实、受限的服务请求，而不只是检查 Key 是否存在。嵌入、图片理解和重排序同样使用云 API；不配置嵌入 API 时仍可使用关键词检索。MinerU 支持官方 API 和自托管端点。DuckDuckGo 搜索无需 Key，Tavily 需要对应环境变量。搜索、网页抓取和 MinerU 请求由本地 EXE 发起，因此不受 GitHub Pages 浏览器 CORS 限制。

知识库支持导入本地 PDF、文本/Markdown、图片和表格，以及搜索或抓取网页。评估数据集使用 JSON 数组，或包含 `cases` / `items` 的对象；每项至少提供 `query`，可通过 `expected` 文档 ID 和 `expected_media` 媒体 ID 计算 Precision@K、Recall@K、MRR、nDCG@K 与媒体召回率。评估结果只允许原子导出到 AutoMemory 的 exports 目录。

基于 **GraphRAG + Agentic RAG** 的智能问答系统。具备**混合路由机制**（根据用户提问自适应选择常规 RAG / GraphRAG / Hybrid），支持**多模态记忆输入**（文本、图片、表格、PDF），集成**高级 RAG 设计模式**（查询重写、混合检索、重排序）。既可作为 **Python 包** `import` 直接使用，也可作为 **FastAPI 服务**对外提供 RESTful API。

---

## 目录

- [核心特性](#核心特性)
- [环境要求](#环境要求)
- [安装](#安装)
- [快速开始（import 即用）](#快速开始import-即用)
- [多模态文档摄入](#多模态文档摄入)
- [高级用法](#高级用法)
- [配置说明](#配置说明)
- [启动 FastAPI 服务](#启动-fastapi-服务)
- [REST API 参考](#rest-api-参考)
- [系统架构](#系统架构)
- [项目结构](#项目结构)
- [测试](#测试)
- [常见问题](#常见问题)

---

## 核心特性

| 特性 | 说明 |
|------|------|
| 🔀 **混合路由** | LLM 意图分类 + 规则双通道，按提问自动路由 Standard RAG / GraphRAG / Hybrid，含置信度阈值与 Fallback |
| 🧠 **Agentic 编排** | 多步推理、查询重写、答案融合、自我反思、对话记忆 |
| 📄 **多模态记忆** | 文本 / 图片（OCR+LLM 描述）/ 表格 / PDF 统一解析与检索 |
| 🔍 **混合检索** | 向量检索 + BM25 关键词 + 图遍历，RRF 倒数排名融合 |
| 📊 **重排序** | BGE 交叉编码器（本地）/ Cohere（API）精排 |
| 🕸️ **GraphRAG** | 实体/关系抽取、知识图谱构建、Louvain 社区检测、图遍历问答 |
| ✏️ **查询重写** | 查询扩展、分解、HyDE 假设性文档、自动策略 |
| 📦 **可打包** | 标准 wheel 包，`pip install` 后 `import agentic_rag` 即用 |
| 🔌 **RESTful API** | FastAPI + Swagger，支持 AutoCode 源码接入 |

---

## 环境要求

- **Python ≥ 3.11**（推荐 3.11+）
- 虚拟环境：`agenticrag`（本仓库已创建）

---

## 安装

### 方式一：安装构建好的 wheel（推荐）

```bash
# 激活虚拟环境（Windows）
agenticrag\Scripts\activate

# 安装 wheel（已生成于 dist/ 目录）
pip install dist/agentic_rag-0.1.0-py3-none-any.whl

# 或安装完整能力（含 torch / chromadb / unstructured 等可选依赖）
pip install "dist/agentic_rag-0.1.0-py3-none-any.whl[all]"
```

### 方式二：源码可编辑安装（开发）

```bash
pip install -e .
```

### 方式三：完整依赖

```bash
# 安装核心依赖
pip install -r requirements.txt

# 安装全部可选依赖（等价于 agentic-rag[all]）
pip install "agentic-rag[all]"
```

> 国内网络建议使用镜像源：`-i https://pypi.tuna.tsinghua.edu.cn/simple --timeout 60`

### 可选依赖组（extras）

| 组名 | 内容 | 说明 |
|------|------|------|
| `all` | 全部可选 | 完整能力 |
| `local-models` | torch, transformers, sentence-transformers | 本地嵌入/重排序模型 |
| `vector-db` | chromadb, qdrant-client | 向量数据库 |
| `pdf` | unstructured[pdf] | PDF 深度解析 |
| `image` | pillow, pytesseract | 图片 OCR |
| `table` | tabula-py, camelot-py, pandas | 表格解析 |
| `dev` | pytest, pytest-asyncio | 测试 |

---

## 快速开始（import 即用）

### 最小示例

```python
from agentic_rag import AgenticRAG

# 1. 初始化（懒加载，首次使用时构建全部组件）
rag = AgenticRAG()

# 2. 摄入文档（自动建图）
rag.ingest_text(
    "人工智能公司深度智能专注于大语言模型研发，其CEO是张伟。"
    "张伟毕业于清华大学计算机系。深度智能与数据科技公司有战略合作关系。",
    metadata={"title": "公司介绍"},
)

# 3. 提问（自动路由：关系类问题走 GraphRAG，事实类问题走 Standard RAG）
ans = rag.query("深度智能公司和数据科技公司有什么关系？")
print(ans["answer"])
print("路由:", ans["route"], "置信度:", ans["confidence"])
```

### 输出示例

```python
{
    "answer": "...",                    # 生成的答案
    "route": "graph",                   # standard / graph / hybrid
    "confidence": 0.85,
    "sources": [...],                   # 引用来源
    "conversation_id": "conv_xxx",
    "latency_ms": 1250,
    "metadata": {...},
}
```

### 指定配置初始化

```python
rag = AgenticRAG(
    llm_api_key="sk-xxx",              # 启用 LLM（意图分类/答案生成/实体抽取）
    embedding_model="BAAI/bge-m3",     # 本地嵌入模型
    vector_db_path="./data/vector_db", # 向量库路径
    graph_db_type="networkx",          # 图存储类型
)
```

---

## 多模态文档摄入

```python
from agentic_rag import AgenticRAG

rag = AgenticRAG()

# 文本
rag.ingest_text("...", metadata={"source": "wiki"})

# 多模态列表（图片为 base64，表格为 CSV/PDF 路径）
result = rag.ingest([
    {"content": "文本内容", "modality": "text", "metadata": {"title": "文档1"}},
    {"content": "<base64图片>", "modality": "image", "metadata": {"title": "图片1"}},
    {"content": "姓名,年龄\n张三,25\n李四,30", "modality": "table", "metadata": {"title": "表格1"}},
    {"content": "file.pdf", "modality": "pdf", "metadata": {"title": "PDF文档"}},
], build_graph=True, chunk_size=512, chunk_overlap=128)

print(result)
# {'status': 'success', 'doc_count': 4, 'chunk_count': N, 'media_count': M, 'reference_count': K, 'graph_stats': {...}, 'message': '...'}
```

| 模态 | content 格式 | 解析方式 |
|------|--------------|----------|
| `text` | 纯文本 | 直接分块 |
| `image` | base64 / 文件路径 | OCR + LLM 图像描述 |
| `table` | CSV 文本 / PDF 字节 | Camelot / Tabula 解析 |
| `pdf` | 文件路径 | Unstructured 深度解析（文本+表格+图片） |

### 多模态检索（基于知识图谱的图片/表格引用，RAG-Anything 风格）

摄入文档时，系统会：

1. **记录引用位置**：检测每个分块中的「图1 / Figure 1 / 表2 / Table 2」引用（`MediaRef`：media_id、label、page、offset）。
2. **构建媒体引用图**：在图存储中建立 `文本块 --references--> 图片/表格` 边（NetworkX / Neo4j）。
3. **存储媒体资产**：抽取的页面图片 / 表格文本存入 `MediaRegistry`（`media_store.py`）。

查询时启用 `enable_multimodal=True`（或环境变量 `AGR_ENABLE_MULTIMODAL_RETRIEVAL=true`），命中的文本块会通过引用图自动附带关联的图片/表格；图片由 **VLM（视觉语言模型）** 描述后交给 LLM，表格直接以文本进入上下文。

VLM 配置（环境变量，或 `POST /api/v1/config/vlm` 在线保存）：

```bash
AGR_VLM_MODEL=gpt-4o            # 例如 gpt-4o / qwen-vl-max / glm-4v
AGR_VLM_API_KEY=sk-...
AGR_VLM_BASE_URL=               # OpenAI 兼容 Base URL
```

未配置 VLM 时，多模态检索仍会返回图片/表格引用（只是没有图片描述）。Python 包提供 `AgenticRAG.save_vlm_config(...)`；Web 版在多模态检索命中图片但未配置 VLM 时会弹窗提醒。

**媒体内存管理** — 抽取的图片 base64 数据持久化到 `AGR_MEDIA_STORE_PATH`（`./data/media/media_registry.json`），内存中仅保留至 `AGR_MEDIA_MAX_MEMORY_MB`（默认 512 MB）上限。超出上限后，较早的图片数据会从内存卸载，按需从磁盘懒加载，避免超大 PDF 耗尽内存。

---

## 高级用法

### 1. 指定路由模式

```python
ans = rag.query("什么是大语言模型？", mode="standard")   # 强制标准 RAG
ans = rag.query("A和B什么关系？",     mode="graph")       # 强制 GraphRAG
ans = rag.query("分析行业趋势",       mode="hybrid")      # 强制混合
ans = rag.query("随便问问",           mode="auto")        # 自动路由（默认）
```

### 2. 多轮对话

```python
ans1 = rag.query("深度智能是做什么的？", conversation_id="conv_1")
ans2 = rag.query("它的CEO是谁？", conversation_id="conv_1")  # 带上文
rag.clear_history("conv_1")
```

### 3. 底层组件直用

```python
from agentic_rag import (
    build_orchestrator, HybridRouter, HybridRetriever,
    GraphRAGEngine, StandardRAGEngine, VectorStoreFactory, GraphStoreFactory,
)

# 构建完整编排器
orch = build_orchestrator()

# 单独使用混合路由器
router = HybridRouter(llm_client=None)
decision = router.route("A公司和B公司有什么关系？")
print(decision.route, decision.confidence)

# 单独使用 GraphRAG 引擎
graph_rag = GraphRAGEngine(graph_store=GraphStoreFactory.create("networkx"))
graph_rag.build_graph_from_documents([{"id": "d1", "content": "..."}])
result = graph_rag.query("...")
```

### 4. 查询重写 / 混合检索 / 重排序

```python
from agentic_rag import QueryRewriter, HybridRetriever, RerankerFactory, EmbedderFactory

rewriter = QueryRewriter(llm_client=None)
rw = rewriter.rewrite("深度智能公司的发展如何？", strategy="auto")
print(rw.variants, rw.sub_queries)          # 改写版本 / 子问题

retriever = HybridRetriever(vector_store=..., graph_store=..., embedder=...)
docs = retriever.retrieve("深度智能公司", top_k=20)   # 向量+BM25+图 融合

reranker = RerankerFactory.create("bge", device="cpu")
ranked = reranker.rerank("深度智能公司", docs, top_k=5)
```

---

## 配置说明

通过环境变量或 `.env` 文件配置（前缀 `AGR_`）。复制 `.env.example` 为 `.env` 后修改：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `AGR_LLM_API_KEY` | - | OpenAI API Key（设置后启用完整 LLM 能力） |
| `AGR_LLM_MODEL` | `gpt-4o-mini` | LLM 模型 |
| `AGR_LLM_BASE_URL` | - | LLM 服务地址（可指向本地 vLLM/Ollama） |
| `AGR_EMBEDDING_MODEL` | `BAAI/bge-m3` | 嵌入模型（本地 BGE 或 OpenAI） |
| `AGR_EMBEDDING_DEVICE` | `cpu` | 嵌入模型设备 |
| `AGR_RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | 重排序模型 |
| `AGR_VECTOR_DB_TYPE` | `chroma` | 向量库：`chroma` / `qdrant` |
| `AGR_VECTOR_DB_PATH` | `./data/vector_db` | ChromaDB 持久化路径 |
| `AGR_GRAPH_DB_TYPE` | `networkx` | 图库：`networkx` / `neo4j` |
| `AGR_NEO4J_URI` | - | Neo4j 连接（使用 Neo4j 时配置） |
| `AGR_API_HOST` / `AGR_API_PORT` | `0.0.0.0` / `8000` | 服务地址 |
| `AGR_API_KEY` | - | API 访问密钥（可选，设置后需 Header 认证） |
| `AGR_ROUTER_CONFIDENCE_THRESHOLD` | `0.6` | 路由置信度阈值 |
| `AGR_ENABLE_FALLBACK` | `true` | 失败自动切换路由 |
| `AGR_TOP_K_INITIAL` / `AGR_TOP_K_RERANK` | `20` / `5` | 检索数量 |
| `AGR_CHUNK_SIZE` / `AGR_CHUNK_OVERLAP` | `512` / `128` | 分块参数 |
| `AGR_LOG_LEVEL` / `AGR_LOG_FILE` | `INFO` / `./data/logs/app.log` | 日志 |

> **未配置 LLM / 模型无法下载时**：系统自动降级为规则路由 + 降级生成，核心流程仍可用（如本仓库当前网络环境下验证）。

---

## 启动 FastAPI 服务

```bash
# 方式一
python -m agentic_rag.main

# 方式二
uvicorn agentic_rag.main:app --host 0.0.0.0 --port 8000 --reload
```

访问：
- Swagger 文档：http://localhost:8000/docs
- 健康检查：http://localhost:8000/api/v1/health

---

## REST API 参考

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/health` | 健康检查 |
| POST | `/api/v1/query` | 问答查询 |
| POST | `/api/v1/ingest` | 文档摄入 |
| POST | `/api/v1/feedback` | 提交反馈 |
| GET | `/api/v1/collections` | 列出集合 |
| DELETE | `/api/v1/collections/{name}` | 删除集合 |

### 查询示例

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "深度智能公司和数据科技公司有什么关系？",
    "mode": "auto",
    "top_k": 5,
    "rerank": true,
    "conversation_id": "conv_001"
  }'
```

### 摄入示例

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {"content": "文本内容", "modality": "text", "metadata": {"title": "文档1"}}
    ],
    "build_graph": true,
    "chunk_size": 512,
    "chunk_overlap": 128
  }'
```

### Python 客户端调用

```python
import requests

resp = requests.post("http://localhost:8000/api/v1/query", json={
    "query": "深度智能是做什么的？",
    "mode": "auto",
})
print(resp.json()["answer"])
```

---

## 系统架构

```
┌──────────────────────────────────────────────────────────────┐
│                     API Layer (FastAPI)                       │
│  POST /query  POST /ingest  POST /feedback  GET /health      │
├──────────────────────────────────────────────────────────────┤
│                   Agentic Orchestrator                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐   │
│  │ 意图分类  │ │ 查询重写  │ │ 路径选择  │ │ 答案合成验证  │   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────┘   │
├──────────┬──────────────────┬──────────────────┬─────────────┤
│ Standard │   GraphRAG       │   Multi-Modal    │   Memory    │
│ RAG      │   Engine         │   Parser         │   Store     │
├──────────┴──────────────────┴──────────────────┴─────────────┤
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│  │向量检索   │ │图遍历检索 │ │关键词检索 │ │混合检索   │      │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘      │
├──────────────────────────────────────────────────────────────┤
│  Vector DB (ChromaDB)    │  Graph DB (NetworkX/Neo4j)       │
│  Embedding Models        │  LLM (OpenAI/LiteLLM)            │
│  Reranker Models         │  Document Parsers                │
└──────────────────────────────────────────────────────────────┘
```

---

## 项目结构

```
F:\intern\Agent\AgenticRag\
├── SPEC.md                    # 规格说明书（含打包发布规格）
├── README.md                  # 本使用文档
├── pyproject.toml             # 打包配置（wheel / extras）
├── requirements.txt           # 依赖清单
├── .env.example               # 环境变量示例
├── dist/                      # 构建产物（agentic_rag-0.1.0-py3-none-any.whl）
├── agenticrag/                # 虚拟环境（Python 3.11.2，全部依赖已装）
├── agentic_rag/               # 源码包
│   ├── __init__.py            # 对外导出（import agentic_rag 即用）
│   ├── client.py              # AgenticRAG 高层客户端（Facade）
│   ├── factory.py             # build_orchestrator 组件工厂
│   ├── service.py             # ingest_documents 文档摄入服务
│   ├── main.py                # FastAPI 应用入口
│   ├── config.py              # 配置管理（pydantic-settings）
│   ├── state.py               # 全局状态
│   ├── api/                   # models.py 请求/响应模型, routes.py API 路由
│   ├── core/                  # orchestrator 编排器, hybrid_router 混合路由, query_rewriter 查询重写
│   ├── rag/                   # standard_rag, graph_rag, hybrid_retriever
│   ├── memory/                # multi_modal_parser, media_store, vector_store, graph_store
│   ├── processing/            # chunker, embedders, reranker
│   └── utils/                 # helpers
└── tests/                     # test_router, test_retriever, test_api
```

---

## 测试

```bash
agenticrag\Scripts\python.exe -m pytest tests/ -v
```

当前 13 个测试全部通过（路由分类、RRF 融合、API 端点）。

---

## 常见问题

**Q1：安装后 `import agentic_rag` 报 ModuleNotFoundError？**
确保激活了 `agenticrag` 虚拟环境，且已执行 `pip install dist/agentic_rag-0.1.0-py3-none-any.whl`。

**Q2：没有 LLM API Key 能用吗？**
能。系统自动降级：规则路由（关键词分类）+ 降级答案生成 + 规则实体抽取。配置 `AGR_LLM_API_KEY` 后启用完整能力。

**Q3：BGE 模型下载失败？**
本地嵌入/重排序模型需从 HuggingFace 下载。可配置 `AGR_EMBEDDING_MODEL` 指向已下载的本地模型路径，或改用 OpenAI 嵌入（`AGR_EMBEDDING_MODEL=text-embedding-3-small` + 配置 API Key）。

**Q4：如何换用 Neo4j？**
配置 `AGR_GRAPH_DB_TYPE=neo4j` 及 `AGR_NEO4J_URI/USER/PASSWORD`。

**Q5：如何重新构建 wheel？**
```bash
agenticrag\Scripts\python.exe -m build --wheel --no-isolation
```

---

## 开发计划

- [x] 核心架构设计
- [x] 混合路由机制
- [x] 多模态解析
- [x] 高级检索模式（查询重写 / 混合检索 / 重排序）
- [x] RESTful API
- [x] 打包发布（wheel / import 即用）
- [ ] 流式输出（SSE）
- [ ] 多轮对话记忆增强
- [ ] 用户反馈在线学习
- [ ] 图可视化界面
- [ ] 分布式部署支持
