# Agentic GraphRAG 系统 — 规格说明书

## 1. 项目概述

### 1.1 目标
构建一个基于 **GraphRAG + Agentic RAG** 的智能问答系统，具备**混合路由机制**，能够根据用户提问自适应选择常规 RAG 或 GraphRAG 路径。系统支持**多模态输入**（文本、图片、表格），集成**高级 RAG 设计模式**（重排序、查询重写、混合检索），对外提供 RESTful API，允许 AutoCode 源码接入。

### 1.2 技术栈
| 组件 | 技术选型 |
|------|----------|
| 编程语言 | Python ≥ 3.11 |
| Web 框架 | FastAPI |
| 向量数据库 | ChromaDB（开发）/ Qdrant（生产） |
| 图数据库 | NetworkX（内存）/ Neo4j（生产） |
| 嵌入模型 | OpenAI Embeddings / BGE-M3（多模态） |
| 重排序模型 | BGE-Reranker / Cohere Rerank |
| LLM | OpenAI GPT-4 / 本地 LLM（通过 LiteLLM） |
| 文档解析 | Unstructured.io / LangChain 文档加载器 |
| 虚拟环境 | agenticrag |

### 1.3 核心能力
- **混合路由**：基于意图分类自适应选择检索路径
- **多模态记忆**：文本、图片、表格统一存储与检索
- **查询重写**：对模糊/复杂问题进行分解与改写
- **混合检索**：向量检索 + 关键词检索 + 图遍历
- **重排序**：交叉编码器对候选结果精排
- **GraphRAG**：从文档构建知识图谱，支持图遍历问答
- **Agentic 编排**：多步推理、工具调用、自我反思

---

## 2. 系统架构

```
┌──────────────────────────────────────────────────────────────────┐
│                        API Layer (FastAPI)                       │
│  POST /query  POST /ingest  POST /feedback  GET /health          │
├──────────────────────────────────────────────────────────────────┤
│                       Agentic Orchestrator                       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐   │
│  │ 意图分类  │ │ 查询重写  │ │ 路径选择  │ │ 答案合成与验证   │   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘   │
├──────────┬──────────────────┬──────────────────┬────────────────┤
│ Standard │   GraphRAG       │   Multi-Modal    │   Memory       │
│ RAG      │   Engine         │   Parser         │   Store        │
├──────────┴──────────────────┴──────────────────┴────────────────┤
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │向量检索   │ │图遍历检索 │ │关键词检索 │ │混合检索   │          │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘          │
├──────────────────────────────────────────────────────────────────┤
│  Vector DB (ChromaDB)    │  Graph DB (NetworkX/Neo4j)           │
│  Embedding Models        │  LLM (OpenAI/LiteLLM)                │
│  Reranker Models         │  Document Parsers                    │
└──────────────────────────────────────────────────────────────────┘
```

### 2.1 分层说明

| 层级 | 职责 |
|------|------|
| **API 层** | 接收外部请求，身份验证，请求/响应序列化 |
| **编排层** | 意图识别、路由决策、多步推理、答案合成 |
| **RAG 引擎层** | 标准 RAG、GraphRAG、混合检索的具体实现 |
| **存储层** | 向量存储、图存储、文档解析与嵌入 |

---

## 3. 组件规格

### 3.1 Hybrid Router（混合路由器）

**职责**：根据用户提问，自适应选择 RAG 路径。

**分类维度**：
| 类别 | 路由目标 | 示例 |
|------|----------|------|
| 事实性查询 | Standard RAG | "2024年GDP是多少？" |
| 关系性查询 | GraphRAG | "A公司和B公司有什么关系？" |
| 多跳推理 | GraphRAG | "A公司的CEO曾在哪所学校就读？" |
| 摘要性查询 | Standard RAG | "这篇文档的主要内容是什么？" |
| 比较性查询 | GraphRAG | "产品X和产品Y的优缺点对比" |

**实现方式**：
1. 轻量级分类器（LLM-based 或小模型）
2. 分类结果附带置信度分数
3. 支持 Fallback 机制（主路径失败时切换）

**接口定义**：
```python
class RouteDecision(BaseModel):
    route: Literal["standard", "graph", "hybrid"]
    confidence: float
    reasoning: str
```

### 3.2 Query Rewriter（查询重写器）

**职责**：对用户原始问题进行改写，提升检索质量。

**策略**：
| 策略 | 说明 |
|------|------|
| 查询扩展 | 添加同义词、相关术语 |
| 查询分解 | 将复杂问题拆分为子问题 |
| 查询澄清 | 补充缺失的上下文信息 |
| 假设性文档 | 生成假设性答案用于检索（HyDE） |

**接口**：
```python
class RewrittenQuery(BaseModel):
    original: str
    variants: list[str]          # 多个改写版本
    sub_queries: list[str]       # 分解后的子问题
    strategy: str                # 使用的改写策略
```

### 3.3 Multi-Modal Parser（多模态解析器）

**职责**：解析不同模态的输入，提取结构化信息。

**支持模态**：
| 模态 | 解析方式 | 存储格式 |
|------|----------|----------|
| 文本 | LangChain 文档加载器 | 文本块 + 元数据 |
| 图片 | OCR + 图像描述（LLM） | 文本描述 + 向量 |
| 表格 | 表格解析器（Camelot/Tabula） | 结构化数据 + 文本表示 |
| PDF | Unstructured.io | 混合块（文本+表格+图片） |

**输出统一格式**：
```python
class UnifiedDocument(BaseModel):
    doc_id: str
    content: str                # 文本内容
    modality: Literal["text", "image", "table", "mixed"]
    metadata: dict
    embeddings: list[float] | None
    chunks: list[DocumentChunk]
```

### 3.4 Standard RAG Engine（标准 RAG 引擎）

**流程**：
1. 接收查询 → 查询重写 → 向量检索 → 重排序 → 答案生成

**检索参数**：
- top_k: 20（初始检索）
- rerank_top_k: 5（重排序后）
- chunk_size: 512
- chunk_overlap: 128

### 3.5 GraphRAG Engine（图 RAG 引擎）

**流程**：
1. 文档 → 实体抽取 → 关系抽取 → 图构建 → 图存储
2. 查询 → 实体识别 → 图遍历 → 子图检索 → 答案生成

**图构建**：
- 实体：人物、组织、地点、概念、事件
- 关系：属性关系、层级关系、时序关系
- 社区检测：Leiden 算法进行社区划分
- 图摘要：每个社区生成摘要信息

**检索策略**：
| 策略 | 说明 |
|------|------|
| 实体匹配 | 从查询中提取实体，直接匹配图节点 |
| 图遍历 | BFS/DFS 遍历相关子图 |
| 社区检索 | 定位实体所在社区，获取社区摘要 |
| 图+向量混合 | 图遍历结果与向量检索结果融合 |

### 3.6 Hybrid Retriever（混合检索器）

**职责**：融合多种检索结果，提供统一的相关文档列表。

**融合策略**：
```python
class HybridRetriever:
    def retrieve(self, query: str, top_k: int = 20) -> list[ScoredDocument]:
        # 1. 向量检索结果
        vector_results = self.vector_store.similarity_search(query, top_k)
        # 2. 关键词检索结果（BM25）
        keyword_results = self.bm25_retriever.search(query, top_k)
        # 3. 图检索结果（如果路由到 GraphRAG）
        graph_results = self.graph_store.traverse(query, top_k)
        # 4. 结果融合（RRF / 加权融合）
        fused = self.reciprocal_rank_fusion(
            [vector_results, keyword_results, graph_results]
        )
        return fused
```

### 3.7 Re-ranker（重排序器）

**职责**：对检索结果进行精细化排序。

**实现**：
- 使用交叉编码器（Cross-Encoder）对查询-文档对打分
- 支持模型：BGE-Reranker / Cohere Rerank / 本地模型

```python
class Reranker:
    def rerank(self, query: str, documents: list[Document], top_k: int = 5) -> list[ScoredDocument]:
        pairs = [(query, doc.content) for doc in documents]
        scores = self.model.predict(pairs)
        # 按分数降序排列，返回 top_k
```

### 3.8 Memory Store（记忆存储）

**职责**：统一管理向量存储和图存储。

**向量存储**：
- 集合（Collections）按文档源/类型组织
- 支持多模态嵌入（文本嵌入、图像嵌入）
- 元数据过滤

**图存储**：
- 节点：实体 + 文档块
- 边：关系 + 相似度连接
- 属性：实体属性、时间戳、来源

### 3.9 Agentic Orchestrator（智能编排器）

**职责**：协调整个问答流程，支持多步推理。

**能力**：
1. **意图识别**：判断查询类型和复杂度
2. **路径规划**：决定使用哪些工具和检索策略
3. **工具调用**：调用检索器、重写器、生成器
4. **结果融合**：合并多源信息
5. **自我反思**：验证答案质量，必要时重新检索
6. **记忆管理**：维护对话历史

---

## 4. API 规格

### 4.1 端点列表

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/query` | 问答查询 |
| POST | `/api/v1/ingest` | 文档摄入 |
| POST | `/api/v1/feedback` | 反馈提交 |
| GET  | `/api/v1/health` | 健康检查 |
| GET  | `/api/v1/collections` | 列出集合 |
| DELETE | `/api/v1/collections/{name}` | 删除集合 |

### 4.2 Query 请求/响应

**请求**：
```json
{
  "query": "2024年人工智能领域有哪些重大突破？",
  "conversation_id": "conv_001",
  "history": [],
  "mode": "auto",
  "top_k": 5,
  "rerank": true
}
```

**响应**：
```json
{
  "answer": "2024年人工智能领域...",
  "route": "hybrid",
  "confidence": 0.95,
  "sources": [
    {
      "doc_id": "doc_001",
      "content": "...",
      "score": 0.92,
      "modality": "text"
    }
  ],
  "conversation_id": "conv_001",
  "latency_ms": 1250
}
```

### 4.3 Ingest 请求

```json
{
  "documents": [
    {
      "content": "...",           // 文本内容或 base64 编码
      "modality": "text|image|table|pdf",
      "metadata": {
        "source": "file.pdf",
        "title": "文档标题",
        "tags": ["AI", "2024"]
      },
      "collection": "default"
    }
  ],
  "chunk_size": 512,
  "chunk_overlap": 128,
  "build_graph": true
}
```

---

## 5. 数据流

### 5.1 文档摄入流程

```
文档输入 → 多模态解析 → 文档分块 → 嵌入生成
    ↓                                        ↓
图构建（可选）                             向量存储
    ↓
图存储
```

### 5.2 查询处理流程

```
用户查询 → 意图分类 → 查询重写 → 混合检索 → 重排序 → 答案生成
    ↓          ↓          ↓          ↓          ↓          ↓
 历史记录   路由决策   改写版本   多源融合   精排结果   最终答案
```

### 5.3 混合路由详细流程

```
用户查询
    ↓
[意图分类器]
    ├── 事实性查询 → Standard RAG → 向量检索 → 重排序 → 生成
    ├── 关系性查询 → GraphRAG → 实体识别 → 图遍历 → 生成
    ├── 多跳推理   → GraphRAG → 子图检索 → 多步推理 → 生成
    ├── 比较性查询 → Hybrid → 图+向量 → 融合 → 生成
    └── 模糊查询   → Hybrid → 查询重写 → 多路径 → 融合 → 生成
```

---

## 6. 部署规格

### 6.1 环境要求
- Python ≥ 3.11
- 虚拟环境名称：agenticrag
- 依赖管理：requirements.txt + pip

### 6.2 配置管理
```python
# config.py
class Settings:
    # 向量数据库
    vector_db_type: str = "chroma"  # chroma | qdrant
    vector_db_path: str = "./data/vector_db"

    # 图数据库
    graph_db_type: str = "networkx"  # networkx | neo4j
    neo4j_uri: str = ""
    neo4j_user: str = ""
    neo4j_password: str = ""

    # 嵌入模型
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024

    # 重排序模型
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # LLM
    llm_model: str = "gpt-4o"
    llm_api_key: str = ""
    llm_base_url: str = ""

    # 检索参数
    top_k_initial: int = 20
    top_k_rerank: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 128

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_key: str = ""
```

### 6.3 项目目录结构
```
F:\intern\Agent\AgenticRag\
├── SPEC.md                    # 本规格文档
├── requirements.txt           # 依赖清单
├── setup.py                   # 安装脚本
├── README.md                  # 项目说明
├── agentic_rag/
│   ├── __init__.py
│   ├── main.py                # 应用入口
│   ├── config.py              # 配置管理
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py          # API 路由
│   │   └── models.py          # 请求/响应模型
│   ├── core/
│   │   ├── __init__.py
│   │   ├── orchestrator.py    # Agentic 编排器
│   │   ├── hybrid_router.py   # 混合路由器
│   │   └── query_rewriter.py  # 查询重写器
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── standard_rag.py    # 标准 RAG 引擎
│   │   ├── graph_rag.py       # GraphRAG 引擎
│   │   └── hybrid_retriever.py# 混合检索器
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── multi_modal_parser.py  # 多模态解析器
│   │   ├── vector_store.py    # 向量存储
│   │   └── graph_store.py     # 图存储
│   ├── processing/
│   │   ├── __init__.py
│   │   ├── embedders.py       # 嵌入器
│   │   ├── reranker.py        # 重排序器
│   │   └── chunker.py         # 文档分块器
│   └── utils/
│       ├── __init__.py
│       └── helpers.py         # 工具函数
└── tests/
    ├── __init__.py
    ├── test_router.py
    ├── test_retriever.py
    └── test_api.py
```

---

## 7. 质量属性

| 属性 | 要求 |
|------|------|
| 可扩展性 | 支持插件式嵌入模型、LLM、存储后端 |
| 可观测性 | 请求日志、链路追踪、性能指标 |
| 容错性 | 单路径失败自动 Fallback |
| 安全性 | API Key 认证、输入校验 |
| 响应时间 | 简单查询 < 2s，复杂查询 < 10s |

---

## 8. 后续迭代方向

- [ ] 流式输出（SSE）
- [ ] 多轮对话记忆
- [ ] 用户反馈在线学习
- [ ] 图可视化界面
- [ ] 分布式部署支持
- [ ] 多语言支持

---

## 9. 打包与发布规格

### 9.1 包结构
- PyPI 包名：`agentic-rag`
- 导入名：`agentic_rag`（源码目录由 `agentic_rag/` 重命名而来）
- 构建后端：setuptools（`pyproject.toml`）
- Python 要求：`>=3.11`

### 9.2 安装方式
```bash
# 方式一：本地可编辑安装（开发）
pip install -e .

# 方式二：构建 wheel 后安装（发布）
pip install build
python -m build
pip install dist/agentic_rag-*.whl

# 方式三：安装全部可选依赖（torch / chromadb / unstructured 等）
pip install agentic-rag[all]
```

### 9.3 对外 API（import 即用）
| 导入语句 | 说明 |
|----------|------|
| `from agentic_rag import AgenticRAG` | 高层客户端 Facade，一行接入完整功能 |
| `from agentic_rag import AgenticOrchestrator, QueryRequest, QueryResponse` | 编排器与数据模型 |
| `from agentic_rag.core.hybrid_router import HybridRouter` | 混合路由器 |
| `from agentic_rag.rag.hybrid_retriever import HybridRetriever` | 混合检索器 |
| `from agentic_rag.rag.graph_rag import GraphRAGEngine` | GraphRAG 引擎 |
| `from agentic_rag.rag.standard_rag import StandardRAGEngine` | 标准 RAG 引擎 |
| `from agentic_rag.memory.vector_store import VectorStoreFactory` | 向量存储工厂 |
| `from agentic_rag.memory.graph_store import GraphStoreFactory` | 图存储工厂 |
| `from agentic_rag.processing.embedders import EmbedderFactory` | 嵌入器工厂 |
| `from agentic_rag.processing.reranker import RerankerFactory` | 重排序器工厂 |
| `from agentic_rag.memory.multi_modal_parser import MultiModalParser` | 多模态解析器 |

### 9.4 组件工厂与高层服务
| 模块 | 职责 |
|------|------|
| `agentic_rag/factory.py` | `build_orchestrator()` 同步构建完整编排器（可传入配置对象） |
| `agentic_rag/service.py` | `ingest_documents()` 文档摄入服务（解析→分块→嵌入→入库→建图） |
| `agentic_rag/client.py` | `AgenticRAG` 高层 Facade（懒加载初始化，提供 query/ingest/health 等） |

### 9.5 可选依赖组（extras）
| 组名 | 内容 |
|------|------|
| `all` | 全部可选依赖（等价于 local-models + pdf + vector-db + image） |
| `local-models` | torch, torchvision, transformers, sentence-transformers |
| `pdf` | unstructured[pdf] |
| `vector-db` | chromadb, qdrant-client |
| `image` | pillow, pytesseract |
| `table` | tabula-py, camelot-py, pandas |
| `dev` | pytest, pytest-asyncio |

### 9.6 依赖策略
- `[project.dependencies]` 仅包含核心轻量依赖（fastapi, pydantic, openai, networkx, numpy, loguru, rank-bm25, jieba 等），保证 `pip install agentic-rag` 快速可用（降级模式）
- 重型依赖（torch/transformers/chromadb/unstructured）放入 extras，通过 `pip install agentic-rag[all]` 启用完整能力