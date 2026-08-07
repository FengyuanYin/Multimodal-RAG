# -*- coding: utf-8 -*-
"""API 端点测试

注意：不进入 TestClient 上下文管理器，避免触发 lifespan 中的 build_orchestrator
（无 LLM Key 时会尝试下载本地嵌入模型，导致测试卡在网络）。
"""
from fastapi.testclient import TestClient

from agentic_rag import state
from agentic_rag.main import app

client = TestClient(app)


def test_health_initializing():
    state.set_orchestrator(None)
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] in ("healthy", "initializing")


def test_query_returns_503_when_not_initialized():
    state.set_orchestrator(None)
    r = client.post("/api/v1/query", json={"query": "你好"})
    assert r.status_code == 503


def test_delete_collection_not_found():
    state.set_orchestrator(None)
    r = client.delete("/api/v1/collections/nonexistent")
    assert r.status_code == 404


def test_ingest_validates_min_length():
    state.set_orchestrator(None)
    r = client.post("/api/v1/ingest", json={"documents": []})
    assert r.status_code == 422


def test_query_validates_empty_query():
    state.set_orchestrator(None)
    r = client.post("/api/v1/query", json={"query": ""})
    assert r.status_code == 422
