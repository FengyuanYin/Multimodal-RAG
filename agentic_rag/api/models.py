"""
API 数据模型模块
==============
定义请求和响应的 Pydantic 模型。
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


# ── 查询相关 ──

class QueryRequest(BaseModel):
    """查询请求"""
    query: str = Field(..., min_length=1, max_length=10000, description="用户查询")
    conversation_id: Optional[str] = Field(None, description="对话 ID")
    history: Optional[List[Dict[str, str]]] = Field(None, description="对话历史")
    mode: Literal["auto", "standard", "graph", "hybrid"] = Field("auto", description="路由模式")
    top_k: int = Field(5, ge=1, le=50, description="返回结果数量")
    rerank: bool = Field(True, description="是否使用重排序")
    stream: bool = Field(False, description="是否流式输出")
    enable_multimodal: Optional[bool] = Field(None, description="是否启用多模态检索（None=使用系统配置）")


class MediaItem(BaseModel):
    """媒体资产项（图片/表格）"""
    id: str = Field("", description="媒体 ID")
    doc_id: str = Field("", description="所属文档 ID")
    type: str = Field("image", description="媒体类型：image | table")
    page: int = Field(1, description="页码")
    label: str = Field("", description="标签，如图1 / 表2")
    caption: str = Field("", description="说明/文本表示")
    refs: List[Dict[str, Any]] = Field(default_factory=list, description="引用位置列表")
    data: Optional[str] = Field(None, description="图片 base64 或表格文本（可选）")


class SourceItem(BaseModel):
    """来源项"""
    doc_id: str = Field("", description="文档 ID")
    content: str = Field("", description="内容片段")
    score: float = Field(0.0, description="相关度分数")
    modality: str = Field("text", description="模态类型")
    media_refs: List[Dict[str, Any]] = Field(default_factory=list, description="引用的图片/表格位置")
    media: Optional[MediaItem] = Field(None, description="关联的媒体资产（图片/表格）")


class QueryResponse(BaseModel):
    """查询响应"""
    answer: str = Field(..., description="生成的答案")
    route: str = Field("auto", description="实际使用的路由")
    confidence: float = Field(0.0, description="置信度")
    sources: List[SourceItem] = Field(default_factory=list, description="信息来源")
    conversation_id: Optional[str] = Field(None, description="对话 ID")
    latency_ms: float = Field(0.0, description="处理耗时（毫秒）")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="额外元数据")


# ── 文档摄入相关 ──

class DocumentItem(BaseModel):
    """文档项"""
    content: str = Field(..., description="文档内容（文本或 base64 编码）")
    modality: Literal["text", "image", "table", "pdf", "mixed"] = Field("text", description="模态类型")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    collection: str = Field("default", description="集合名称")


class IngestRequest(BaseModel):
    """文档摄入请求"""
    documents: List[DocumentItem] = Field(..., min_length=1, max_length=100, description="文档列表")
    chunk_size: int = Field(512, ge=64, le=2048, description="分块大小")
    chunk_overlap: int = Field(128, ge=0, le=512, description="分块重叠")
    build_graph: bool = Field(True, description="是否构建知识图谱")


class IngestResponse(BaseModel):
    """文档摄入响应"""
    status: str = Field("success", description="状态")
    doc_count: int = Field(0, description="文档数量")
    chunk_count: int = Field(0, description="分块数量")
    media_count: int = Field(0, description="媒体资产数量（图片/表格）")
    reference_count: int = Field(0, description="图/表引用数量")
    graph_stats: Optional[Dict[str, Any]] = Field(None, description="图构建统计")
    message: str = Field("", description="消息")


# ── 反馈相关 ──

class FeedbackRequest(BaseModel):
    """反馈请求"""
    conversation_id: str = Field(..., description="对话 ID")
    query: str = Field(..., description="原始查询")
    answer: str = Field(..., description="生成的答案")
    rating: int = Field(..., ge=1, le=5, description="评分 1-5")
    feedback: Optional[str] = Field(None, description="反馈文本")


class FeedbackResponse(BaseModel):
    """反馈响应"""
    status: str = Field("success", description="状态")
    message: str = Field("感谢您的反馈！", description="消息")


# ── 健康检查 ──

class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field("healthy", description="服务状态")
    version: str = Field("0.1.0", description="版本")
    vector_store: str = Field("", description="向量存储状态")
    graph_store: str = Field("", description="图存储状态")
    llm: str = Field("", description="LLM 状态")


# ── 集合管理 ──

class CollectionListResponse(BaseModel):
    """集合列表响应"""
    collections: List[str] = Field(default_factory=list, description="集合列表")
    total: int = Field(0, description="总数")


# ── VLM 配置（多模态检索） ──

class VLMSettingsRequest(BaseModel):
    """VLM 配置请求"""
    provider: str = Field("openai", description="提供商：openai | litellm")
    model: str = Field("", description="VLM 模型名，如 gpt-4o / qwen-vl-max")
    api_key: Optional[str] = Field(None, description="API Key（留空则保持原值）")
    base_url: Optional[str] = Field(None, description="Base URL（OpenAI 兼容）")


class VLMSettingsResponse(BaseModel):
    """VLM 配置响应（脱敏）"""
    provider: str = Field("openai", description="提供商")
    model: str = Field("", description="模型名")
    base_url: str = Field("", description="Base URL")
    configured: bool = Field(False, description="是否已配置")


class ConfigResponse(BaseModel):
    """系统配置状态"""
    version: str = Field("", description="版本")
    llm_configured: bool = Field(False, description="LLM 是否已配置")
    vlm: VLMSettingsResponse = Field(default_factory=VLMSettingsResponse, description="VLM 配置状态")
    enable_multimodal_retrieval: bool = Field(False, description="是否启用多模态检索")
    media_count: int = Field(0, description="已注册媒体资产数量")