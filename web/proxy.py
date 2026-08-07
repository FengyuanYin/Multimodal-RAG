"""
PDF Chat 可选同源代理服务器
==========================
纯前端页面直连 LLM API 时，部分提供商不支持浏览器 CORS，会被浏览器拦截。
若你自托管本服务，可启动该代理，让前端通过同源路径转发请求，绕过 CORS。

启动方式：
    agenticrag\\Scripts\\python.exe -m uvicorn web.proxy:app --host 0.0.0.0 --port 8000

安全说明：
    - 代理本身不存储任何 API Key。
    - Key 由浏览器放在 X-API-Key 请求头中，代理仅原样转发给用户填写的 base_url。
    - 生产环境请自行加 HTTPS 与访问控制。
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx

app = FastAPI(title="PDF Chat Proxy")

# 允许任意来源：代理本身不存 Key，CORS 放开是为了让任意前端页面可用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_TIMEOUT = httpx.Timeout(120.0, connect=10.0)


async def _forward(request: Request, path: str):
    """将请求转发到客户端指定的 base_url，保持原始 body 与鉴权头"""
    body = await request.body()
    # 读取前端传入的目标地址与模型配置
    try:
        import json
        payload = json.loads(body) if body else {}
    except Exception:
        payload = {}

    base_url = payload.pop("base_url", None) or payload.pop("baseUrl", None)
    if not base_url:
        return JSONResponse({"error": "缺少 base_url"}, status_code=400)

    # 客户端 API Key：优先取请求头，其次取 body
    api_key = request.headers.get("X-API-Key") or payload.pop("api_key", None)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    url = f"{base_url.rstrip('/')}/{path}"
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.request(
                request.method,
                url,
                content=json.dumps(payload).encode("utf-8"),
                headers=headers,
            )
        return JSONResponse(
            status_code=resp.status_code,
            content=resp.json() if resp.content else {"error": "empty response"},
        )
    except httpx.HTTPError as e:
        return JSONResponse({"error": f"代理请求失败: {e}"}, status_code=502)


@app.post("/proxy/chat/completions")
async def proxy_chat(request: Request):
    return await _forward(request, "chat/completions")


@app.post("/proxy/embeddings")
async def proxy_embeddings(request: Request):
    return await _forward(request, "embeddings")


@app.get("/proxy/health")
async def proxy_health():
    return {"status": "ok"}
