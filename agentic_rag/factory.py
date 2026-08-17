"""
组件工厂模块
============
同步构建完整的 Agentic 编排器及其所有依赖组件。
供 FastAPI 启动流程与 AgenticRAG 高层客户端复用。
"""

from loguru import logger

from agentic_rag.config import settings as default_settings


def build_orchestrator(settings_obj=None):
    """
    构建编排器及所有依赖组件

    Args:
        settings_obj: 可选配置对象（默认使用全局 settings）

    Returns:
        AgenticOrchestrator: 配置完成的编排器实例
    """
    from agentic_rag.core.orchestrator import AgenticOrchestrator
    from agentic_rag.core.hybrid_router import HybridRouter
    from agentic_rag.core.query_rewriter import QueryRewriter
    from agentic_rag.rag.standard_rag import StandardRAGEngine
    from agentic_rag.rag.graph_rag import GraphRAGEngine, EntityRelationExtractor
    from agentic_rag.rag.hybrid_retriever import HybridRetriever
    from agentic_rag.memory.vector_store import VectorStoreFactory
    from agentic_rag.memory.graph_store import GraphStoreFactory
    from agentic_rag.processing.embedders import EmbedderFactory
    from agentic_rag.processing.reranker import RerankerFactory
    from agentic_rag.memory.knowledge_repository import KnowledgeRepository

    cfg = settings_obj or default_settings

    # 1. LLM 客户端
    llm_client = None
    if getattr(cfg, "llm_api_key", None):
        try:
            from openai import OpenAI
            llm_client = OpenAI(
                api_key=cfg.llm_api_key,
                base_url=getattr(cfg, "llm_base_url", None),
            )
            logger.info(f"LLM 客户端初始化: {cfg.llm_model}")
        except Exception as e:
            logger.warning(f"LLM 客户端初始化失败: {e}")

    # 1.1 VLM 客户端（视觉语言模型，多模态检索/图片描述）
    vlm_client = None
    if getattr(cfg, "vlm_api_key", None):
        try:
            from openai import OpenAI
            vlm_client = OpenAI(
                api_key=cfg.vlm_api_key,
                base_url=getattr(cfg, "vlm_base_url", None) or getattr(cfg, "llm_base_url", None),
            )
            logger.info(f"VLM 客户端初始化: {cfg.vlm_model}")
        except Exception as e:
            logger.warning(f"VLM 客户端初始化失败: {e}")
    elif getattr(cfg, "llm_api_key", None):
        # 未单独配置 VLM 时，回退使用 LLM 客户端（部分模型支持图像输入）
        vlm_client = llm_client
        logger.info(f"VLM 客户端回退使用 LLM 客户端: {cfg.llm_model}")

    # 2. 嵌入器
    embedder = None
    try:
        embedder = EmbedderFactory.create(
            provider="bge" if "bge" in cfg.embedding_model else "openai",
            model_name=cfg.embedding_model,
            device=cfg.embedding_device,
            dim=cfg.embedding_dim,
            api_key=getattr(cfg, "llm_api_key", None),
        )
        logger.info(f"嵌入器初始化: {cfg.embedding_model}")
    except Exception as e:
        logger.warning(f"嵌入器初始化失败: {e}")

    # 3. 向量存储
    vector_store = None
    try:
        vector_store = VectorStoreFactory.create(
            db_type=cfg.vector_db_type,
            collection_name=cfg.milvus_collection,
            embedding_dim=cfg.embedding_dim,
            uri=cfg.milvus_uri,
            database=cfg.milvus_database,
            token=cfg.milvus_token,
            timeout_seconds=cfg.milvus_timeout_seconds,
        )
        logger.info(f"向量存储初始化: {cfg.vector_db_type}")
    except Exception as e:
        logger.warning(f"向量存储初始化失败: {e}")

    # 4. 图存储
    graph_store = None
    try:
        graph_store = GraphStoreFactory.create(
            db_type=cfg.graph_db_type,
            uri=getattr(cfg, "neo4j_uri", None),
            user=getattr(cfg, "neo4j_user", None),
            password=getattr(cfg, "neo4j_password", None),
        )
        logger.info(f"图存储初始化: {cfg.graph_db_type}")
    except Exception as e:
        logger.warning(f"图存储初始化失败: {e}")

    # 5. 重排序器
    reranker = None
    try:
        reranker = RerankerFactory.create(
            provider="bge",
            model_name=cfg.reranker_model,
            device=cfg.reranker_device,
        )
        logger.info(f"重排序器初始化: {cfg.reranker_model}")
    except Exception as e:
        logger.warning(f"重排序器初始化失败: {e}")

    # 6. SQLite 主知识库与混合检索器
    knowledge_repository = KnowledgeRepository(
        getattr(cfg, "knowledge_db_path", "./data/knowledge/knowledge.db")
    )
    hybrid_retriever = HybridRetriever(
        vector_store=vector_store,
        graph_store=graph_store,
        embedder=embedder,
        knowledge_repository=knowledge_repository,
    )
    # 媒体资产注册表（RAG-Anything 风格：图片/表格数据 + 引用位置）
    # 自动持久化到磁盘 + 内存上限（超出从磁盘懒加载），避免超大 PDF base64 全部驻留内存
    from agentic_rag.memory.media_store import MediaRegistry
    media_store = MediaRegistry(
        persist_path=getattr(cfg, "media_store_path", "./data/media/media_registry.json"),
        auto_save=True,
        max_memory_bytes=getattr(cfg, "media_max_memory_mb", 512) * 1024 * 1024,
    )
    media_store.load()
    hybrid_retriever.media_store = media_store
    # 从持久化文件重建 BM25 索引（服务重启 / 新进程时保留已摄入文档）
    try:
        loaded = hybrid_retriever.rebuild_from_repository()
        if not loaded:
            hybrid_retriever.load_persisted_index()
    except Exception as e:
        logger.warning(f"加载持久化索引失败: {e}")

    # 7. 混合路由器
    hybrid_router = HybridRouter(
        llm_client=llm_client,
        confidence_threshold=cfg.router_confidence_threshold,
        enable_fallback=cfg.enable_fallback,
        llm_model=cfg.llm_model,
    )

    # 8. 查询重写器
    query_rewriter = QueryRewriter(llm_client=llm_client, llm_model=cfg.llm_model)

    # 9. 标准 RAG 引擎
    standard_rag = StandardRAGEngine(
        retriever=hybrid_retriever,
        reranker=reranker,
        embedder=embedder,
        llm_client=llm_client,
        llm_model=cfg.llm_model,
        top_k_rerank=cfg.top_k_rerank,
        vlm_client=vlm_client,
        vlm_model=getattr(cfg, "vlm_model", None),
    )

    # 10. GraphRAG 引擎
    extractor = EntityRelationExtractor(llm_client=llm_client, llm_model=cfg.llm_model)
    graph_rag = GraphRAGEngine(
        graph_store=graph_store,
        vector_store=vector_store,
        embedder=embedder,
        reranker=reranker,
        llm_client=llm_client,
        llm_model=cfg.llm_model,
        extractor=extractor,
    )

    # 11. 编排器
    orchestrator = AgenticOrchestrator(
        router=hybrid_router,
        query_rewriter=query_rewriter,
        standard_rag=standard_rag,
        graph_rag=graph_rag,
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        llm_client=llm_client,
        llm_model=cfg.llm_model,
        vlm_client=vlm_client,
        vlm_model=getattr(cfg, "vlm_model", None),
        media_store=media_store,
        enable_multimodal=getattr(cfg, "enable_multimodal_retrieval", False),
    )
    orchestrator.knowledge_repository = knowledge_repository

    logger.info("所有组件初始化完成")
    return orchestrator
