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


# ── VLM 配置接口（多模态检索） ──

def test_get_config_returns_vlm_status():
    state.set_orchestrator(None)
    r = client.get("/api/v1/config")
    assert r.status_code == 200
    body = r.json()
    assert "vlm" in body
    assert "configured" in body["vlm"]
    assert "enable_multimodal_retrieval" in body
    assert "media_count" in body


def test_get_vlm_config():
    state.set_orchestrator(None)
    r = client.get("/api/v1/config/vlm")
    assert r.status_code == 200
    body = r.json()
    assert "configured" in body
    assert "model" in body
    # 默认未配置时不应泄露密钥
    assert "api_key" not in body


def test_save_vlm_config_updates_settings_and_env():
    import tempfile, os
    state.set_orchestrator(None)
    from agentic_rag.config import settings

    tmpdir = tempfile.mkdtemp(prefix="agr_vlm_test_")
    env_file = os.path.join(tmpdir, ".env")
    try:
        result = settings.save_vlm_config(
            provider="openai",
            model="qwen-vl-max",
            api_key="sk-test-123",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            env_path=env_file,
        )
        assert result["configured"] is True
        assert result["model"] == "qwen-vl-max"
        content = open(env_file, encoding="utf-8").read()
        assert "AGR_VLM_MODEL=qwen-vl-max" in content
        assert "AGR_VLM_API_KEY=sk-test-123" in content
    finally:
        # 恢复默认配置，避免影响其他测试
        settings.save_vlm_config(model="gpt-4o-mini", api_key=None, base_url=None, env_path=env_file)
        try:
            os.remove(env_file)
            os.rmdir(tmpdir)
        except OSError:
            pass
