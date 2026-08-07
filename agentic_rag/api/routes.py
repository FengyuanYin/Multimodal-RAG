"""
API 路由模块
==========
定义 FastAPI 路由，提供 RESTful API 接口。
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, Header
from loguru import logger

from agentic_rag.api.models import (
    QueryRequest, QueryResponse, SourceItem, MediaItem,
    IngestRequest, IngestResponse,
    FeedbackRequest, FeedbackResponse,
    HealthResponse, CollectionListResponse,
    VLMSettingsRequest, VLMSettingsResponse, ConfigResponse,
)
from agentic_rag.core.orchestrator import AgenticOrchestrator, QueryRequest as OrchestratorRequest
from agentic_rag.state import get_orchestrator

# 创建路由器
router = APIRouter(prefix="/api/v1")


async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """API Key 验证（可选）"""
    from agentic_rag.config import settings
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="无效的 API Key")
    return True


# ── 健康检查 ──

@router.get("/health", response_model=HealthResponse, tags=["系统"])
async def health_check():
    """健康检查端点"""
    from agentic_rag.config import settings
    orch = get_orchestrator()
    if orch is None:
        return HealthResponse(status="initializing", version=settings.app_version)

    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        vector_store="connected" if (orch.hybrid_retriever and orch.hybrid_retriever.vector_store) else "not_configured",
        graph_store="connected" if (orch.graph_rag and orch.graph_rag.graph_store) else "not_configured",
        llm="connected" if orch.llm_client else "not_configured",
    )


# ── 查询 ──

@router.post("/query", response_model=QueryResponse, tags=["查询"])
async def query(request: QueryRequest, _=Depends(verify_api_key)):
    """问答查询端点"""
    orch = get_orchestrator()
    if orch is None:
        raise HTTPException(status_code=503, detail="系统正在初始化，请稍后重试")

    # 转换为编排器请求
    orch_request = OrchestratorRequest(
        query=request.query,
        conversation_id=request.conversation_id,
        history=request.history,
        mode=request.mode,
        top_k=request.top_k,
        rerank=request.rerank,
        stream=request.stream,
        enable_multimodal=request.enable_multimodal,
    )

    # 执行查询
    result = orch.query(orch_request)

    # 转换为 API 响应
    return QueryResponse(
        answer=result.answer,
        route=result.route,
        confidence=result.confidence,
        sources=[
            SourceItem(
                doc_id=s.get("doc_id", ""),
                content=s.get("content", "")[:500],
                score=s.get("score", 0.0),
                modality=s.get("modality", "text"),
                media_refs=s.get("media_refs", []),
                media=MediaItem(**s["media"]) if s.get("media") else None,
            )
            for s in (result.sources or [])
        ],
        conversation_id=result.conversation_id,
        latency_ms=result.latency_ms,
        metadata=result.metadata,
    )


# ── 文档摄入 ──

@router.post("/ingest", response_model=IngestResponse, tags=["文档管理"])
async def ingest(request: IngestRequest, _=Depends(verify_api_key)):
    """文档摄入端点"""
    orch = get_orchestrator()
    if orch is None:
        raise HTTPException(status_code=503, detail="系统正在初始化，请稍后重试")

    from agentic_rag.service import ingest_documents

    result = ingest_documents(
        orchestrator=orch,
        documents=[
            {
                "content": d.content,
                "modality": d.modality,
                "metadata": d.metadata,
                "collection": d.collection,
            }
            for d in request.documents
        ],
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
        build_graph=request.build_graph,
    )

    return IngestResponse(
        status=result["status"],
        doc_count=result["doc_count"],
        chunk_count=result["chunk_count"],
        media_count=result.get("media_count", 0),
        reference_count=result.get("reference_count", 0),
        graph_stats=result["graph_stats"],
        message=result["message"],
    )


# ── 反馈 ──

@router.post("/feedback", response_model=FeedbackResponse, tags=["系统"])
async def feedback(request: FeedbackRequest, _=Depends(verify_api_key)):
    """提交反馈"""
    logger.info(f"收到反馈: conv={request.conversation_id}, rating={request.rating}")
    return FeedbackResponse(
        status="success",
        message="感谢您的反馈！",
    )


# ── 集合管理 ──

@router.get("/collections", response_model=CollectionListResponse, tags=["文档管理"])
async def list_collections(_=Depends(verify_api_key)):
    """列出所有集合"""
    orch = get_orchestrator()
    if orch and orch.hybrid_retriever and orch.hybrid_retriever.vector_store:
        collections = orch.hybrid_retriever.vector_store.list_collections()
        return CollectionListResponse(collections=collections, total=len(collections))
    return CollectionListResponse(collections=[], total=0)


@router.delete("/collections/{name}", tags=["文档管理"])
async def delete_collection(name: str, _=Depends(verify_api_key)):
    """删除集合（含其中所有向量数据）"""
    orch = get_orchestrator()
    if orch and orch.hybrid_retriever and orch.hybrid_retriever.vector_store:
        store = orch.hybrid_retriever.vector_store
        collections = store.list_collections()
        if name not in collections:
            raise HTTPException(status_code=404, detail=f"集合 {name} 不存在")
        ok = store.delete_collection(name)
        if not ok:
            raise HTTPException(status_code=500, detail=f"集合 {name} 删除失败")
        return {"status": "success", "message": f"集合 {name} 已删除"}
    raise HTTPException(status_code=404, detail="集合不存在")


# ── VLM 配置与系统配置状态 ──

@router.get("/config", response_model=ConfigResponse, tags=["配置"])
async def get_config(_=Depends(verify_api_key)):
    """获取系统配置状态（LLM/VLM/多模态检索开关）"""
    from agentic_rag.config import settings
    orch = get_orchestrator()
    media_count = 0
    if orch and getattr(orch, "media_store", None):
        media_count = orch.media_store.count
    vlm = settings.vlm_config_dict()
    return ConfigResponse(
        version=settings.app_version,
        llm_configured=bool(settings.llm_api_key),
        vlm=VLMSettingsResponse(
            provider=vlm["provider"],
            model=vlm["model"],
            base_url=vlm["base_url"],
            configured=vlm["configured"],
        ),
        enable_multimodal_retrieval=bool(getattr(orch, "enable_multimodal", False) if orch else settings.enable_multimodal_retrieval),
        media_count=media_count,
    )


@router.get("/config/vlm", response_model=VLMSettingsResponse, tags=["配置"])
async def get_vlm_config(_=Depends(verify_api_key)):
    """获取 VLM 配置（脱敏）"""
    from agentic_rag.config import settings
    vlm = settings.vlm_config_dict()
    return VLMSettingsResponse(
        provider=vlm["provider"],
        model=vlm["model"],
        base_url=vlm["base_url"],
        configured=vlm["configured"],
    )


@router.post("/config/vlm", response_model=VLMSettingsResponse, tags=["配置"])
async def save_vlm_config(request: VLMSettingsRequest, _=Depends(verify_api_key)):
    """保存 VLM 配置（写入 .env，重启后仍生效；同时尝试热更新当前编排器）"""
    from agentic_rag.config import settings
    result = settings.save_vlm_config(
        provider=request.provider or None,
        model=request.model or None,
        api_key=request.api_key,
        base_url=request.base_url,
    )

    # 热更新：尝试重建 VLM 客户端并挂到编排器/标准 RAG 引擎
    orch = get_orchestrator()
    if orch is not None:
        try:
            vlm_client = None
            if settings.vlm_api_key:
                from openai import OpenAI
                vlm_client = OpenAI(
                    api_key=settings.vlm_api_key,
                    base_url=settings.vlm_base_url or settings.llm_base_url,
                )
            elif settings.llm_api_key:
                vlm_client = orch.llm_client
            orch.vlm_client = vlm_client
            orch.vlm_model = settings.vlm_model
            if orch.standard_rag:
                orch.standard_rag.vlm_client = vlm_client
                orch.standard_rag.vlm_model = settings.vlm_model
            logger.info("VLM 客户端已热更新")
        except Exception as e:
            logger.warning(f"VLM 客户端热更新失败（重启后生效）: {e}")

    return VLMSettingsResponse(**result)