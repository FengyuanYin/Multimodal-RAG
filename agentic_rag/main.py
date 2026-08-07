"""
系统主入口
=========
FastAPI 应用初始化，依赖注入，启动/关闭事件。
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from agentic_rag.config import settings
from agentic_rag.api.routes import router as api_router
from agentic_rag.state import set_orchestrator
from agentic_rag.utils.helpers import setup_logger


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """应用生命周期管理：启动时初始化组件，关闭时清理资源"""
    logger.info("=" * 60)
    logger.info(f"  {settings.app_name} v{settings.app_version}")
    logger.info("=" * 60)

    # ── 启动阶段 ──
    try:
        from agentic_rag.factory import build_orchestrator
        orchestrator = build_orchestrator()
        set_orchestrator(orchestrator)
        logger.info("编排器初始化完成")
    except Exception as e:
        logger.error(f"编排器初始化失败: {e}")
        logger.warning("系统将以降级模式运行（部分功能不可用）")

    yield

    # ── 关闭阶段 ──
    logger.info("正在关闭系统...")
    from agentic_rag.state import get_orchestrator
    orch = get_orchestrator()
    if orch:
        if orch.graph_rag and hasattr(orch.graph_rag.graph_store, 'close'):
            orch.graph_rag.graph_store.close()
        if getattr(orch, "knowledge_repository", None):
            orch.knowledge_repository.close()
        orch.clear_history()
    logger.info("系统已关闭")


def create_app() -> FastAPI:
    """创建并配置 FastAPI 应用"""
    # 配置日志
    setup_logger(settings.log_file, settings.log_level)

    # 创建应用
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Agentic GraphRAG 系统 — 混合路由 + 多模态记忆 + 高级 RAG 模式",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    allowed_origins = [item.strip() for item in settings.allowed_cors_origins.split(",") if item.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "X-API-Key", "Authorization"],
    )

    # 注册路由
    app.include_router(api_router)

    return app


# 创建应用实例
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "agentic_rag.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
