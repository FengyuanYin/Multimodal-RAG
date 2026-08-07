"""
PDF Chat 可选同源代理服务器
=========================
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
import asyncio
import ipaddress
import os
import socket
from urllib.parse import urljoin, urlsplit

app = FastAPI(title="PDF Chat Proxy")

_ALLOWED_ORIGINS = [item.strip() for item in os.getenv(
    "AGR_PROXY_ALLOWED_ORIGINS",
    "https://fengyuanyin.github.io,http://localhost:8000,http://127.0.0.1:8000",
).split(",") if item.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)

_TIMEOUT = httpx.Timeout(120.0, connect=10.0)
_FETCH_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
_MAX_FETCH_BYTES = 2 * 1024 * 1024  # 网页抓取大小上限 2MB
_MAX_REQUEST_BYTES = 10 * 1024 * 1024
_ALLOW_PRIVATE = os.getenv("AGR_PROXY_ALLOW_PRIVATE", "").lower() in {"1", "true", "yes"}
_ALLOWED_TARGET_HOSTS = {item.strip().lower() for item in os.getenv("AGR_PROXY_ALLOWED_HOSTS", "").split(",") if item.strip()}


async def _validate_target(url: str) -> str:
    """拒绝凭据 URL、非 HTTP 协议和解析到内网/保留地址的目标。"""
    parsed = urlsplit(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("目标地址必须是无凭据的 HTTP(S) URL")
    host = parsed.hostname.lower().rstrip(".")
    if _ALLOWED_TARGET_HOSTS and host not in _ALLOWED_TARGET_HOSTS:
        raise ValueError("目标主机不在允许列表中")
    if _ALLOW_PRIVATE:
        return url
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError("目标主机无法解析") from exc
    addresses = {item[4][0].split("%", 1)[0] for item in infos}
    if not addresses:
        raise ValueError("目标主机没有可用地址")
    for value in addresses:
        address = ipaddress.ip_address(value)
        if not address.is_global:
            raise ValueError("禁止访问本地、内网或保留地址")
    return url


async def _forward(request: Request, path: str):
    """将请求转发到客户端指定的 base_url，保持原始 body 与鉴权头"""
    body = await request.body()
    if len(body) > _MAX_REQUEST_BYTES:
        return JSONResponse({"error": "请求体超过 10MB 上限"}, status_code=413)
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
        await _validate_target(url)
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
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except httpx.HTTPError:
        return JSONResponse({"error": "上游代理请求失败"}, status_code=502)


@app.post("/proxy/chat/completions")
async def proxy_chat(request: Request):
    return await _forward(request, "chat/completions")


@app.post("/proxy/embeddings")
async def proxy_embeddings(request: Request):
    return await _forward(request, "embeddings")


# ── Web 搜索与网页抓取（制作知识库） ──


def _is_safe_url(url: str) -> bool:
    """无网络解析的基础 URL 语法校验。"""
    try:
        parsed = urlsplit((url or "").strip())
        return parsed.scheme in {"http", "https"} and bool(parsed.hostname) and not parsed.username and not parsed.password
    except ValueError:
        return False


def _html_to_text(html: str) -> str:
    """简单 HTML → 正文文本（去 script/style/导航，规范化空白）"""
    import re
    from html.parser import HTMLParser

    class TextExtractor(HTMLParser):
        SKIP_TAGS = {"script", "style", "noscript", "svg", "iframe", "nav", "footer", "header", "aside"}

        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.parts = []
            self.skip_depth = 0

        def handle_starttag(self, tag, attrs):
            if tag in self.SKIP_TAGS:
                self.skip_depth += 1
            elif tag in ("p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "pre", "blockquote"):
                self.parts.append("\n")

        def handle_endtag(self, tag):
            if tag in self.SKIP_TAGS and self.skip_depth > 0:
                self.skip_depth -= 1

        def handle_data(self, data):
            if self.skip_depth == 0:
                self.parts.append(data)

    parser = TextExtractor()
    try:
        parser.feed(html)
    except Exception:
        pass
    text = "".join(parser.parts)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_title(html: str) -> str:
    import re
    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()
    return ""


async def _fetch_html(url: str) -> tuple:
    """抓取网页 HTML；返回 (html, final_url)"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }
    current = url
    async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT, follow_redirects=False) as client:
        for _ in range(4):
            await _validate_target(current)
            async with client.stream("GET", current, headers=headers) as resp:
                if resp.is_redirect:
                    location = resp.headers.get("location")
                    if not location:
                        raise ValueError("重定向缺少目标地址")
                    current = urljoin(current, location)
                    continue
                resp.raise_for_status()
                content_type = resp.headers.get("content-type", "").lower()
                if content_type and not any(item in content_type for item in ("text/html", "application/xhtml+xml", "text/plain")):
                    raise ValueError("目标不是可解析的网页内容")
                declared_length = int(resp.headers.get("content-length", "0") or 0)
                if declared_length > _MAX_FETCH_BYTES:
                    raise ValueError("网页内容超过 2MB 上限")
                chunks = []
                total = 0
                async for chunk in resp.aiter_bytes():
                    total += len(chunk)
                    if total > _MAX_FETCH_BYTES:
                        raise ValueError("网页内容超过 2MB 上限")
                    chunks.append(chunk)
                final_url = str(resp.url)
                return b"".join(chunks).decode(resp.encoding or "utf-8", errors="ignore"), final_url
        raise ValueError("网页重定向次数过多")


@app.get("/proxy/web/fetch")
async def proxy_web_fetch(url: str):
    """抓取网页并转换为文本（供前端制作知识库）"""
    if not _is_safe_url(url):
        return JSONResponse({"error": "仅支持 http/https URL"}, status_code=400)
    try:
        html, final_url = await _fetch_html(url)
        title = _extract_title(html) or final_url
        text = _html_to_text(html)
        if not text:
            return JSONResponse({"error": "网页未提取到正文文本（可能为动态渲染页面）"}, status_code=422)
        return JSONResponse({
            "url": final_url,
            "title": title,
            "text": text[:500000],  # 单页最多保留 50 万字符
        })
    except httpx.HTTPError:
        return JSONResponse({"error": "网页抓取上游请求失败"}, status_code=502)
    except Exception as e:
        return JSONResponse({"error": f"网页抓取失败: {e}"}, status_code=502)


@app.post("/proxy/web/search")
async def proxy_web_search(request: Request):
    """Web 搜索：DuckDuckGo（默认，无需 Key）或 Tavily（需 Key）"""
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "请求体必须是 JSON"}, status_code=400)

    query = (payload.get("query") or "").strip()
    if not query:
        return JSONResponse({"error": "缺少 query"}, status_code=400)
    if len(query) > 500:
        return JSONResponse({"error": "query 超过 500 字符上限"}, status_code=413)

    provider = (payload.get("provider") or "duckduckgo").lower()
    api_key = (payload.get("api_key") or "").strip()
    max_results = max(1, min(10, int(payload.get("max_results") or 6)))

    try:
        if provider == "tavily":
            if not api_key:
                return JSONResponse({"error": "Tavily 需要 API Key"}, status_code=400)
            results = await _search_tavily(query, api_key, max_results)
        else:
            results = await _search_duckduckgo(query, max_results)
        return JSONResponse({"results": results, "provider": provider})
    except httpx.HTTPError:
        return JSONResponse({"error": "搜索上游请求失败"}, status_code=502)
    except Exception as e:
        return JSONResponse({"error": f"搜索失败: {e}"}, status_code=502)


async def _search_duckduckgo(query: str, max_results: int) -> list:
    """DuckDuckGo HTML 搜索（免费、无需 Key）"""
    import re
    from urllib.parse import quote, unquote

    url = f"https://html.duckduckgo.com/html/?q={quote(query)}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0"}
    async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT, follow_redirects=True) as client:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        html = resp.text

    results = []
    # DuckDuckGo 结果结构：<a class="result__a" href="...">标题</a> ... <a class="result__snippet">摘要</a>
    for m in re.finditer(
        r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>.*?'
        r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
        html, re.IGNORECASE | re.DOTALL,
    ):
        href, title_html, snippet_html = m.group(1), m.group(2), m.group(3)
        # 解析跳转链接中的 uddg 参数
        uddg = re.search(r"uddg=([^&]+)", href)
        real_url = unquote(uddg.group(1)) if uddg else href
        if not _is_safe_url(real_url):
            continue
        title = re.sub(r"<[^>]+>", "", title_html).strip()
        snippet = re.sub(r"<[^>]+>", "", snippet_html).strip()
        results.append({"title": title, "url": real_url, "snippet": snippet})
        if len(results) >= max_results:
            break
    return results


async def _search_tavily(query: str, api_key: str, max_results: int) -> list:
    """Tavily Search API（专为 RAG 设计，返回正文片段）"""
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": "basic",
    }
    async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", "")[:400],
        }
        for r in (data.get("results") or [])
    ]


# ── MinerU 官方 API 转发（https://mineru.net/api/v4） ──
# 前端将 API Key 放在 X-API-Key 头，代理转发为 Authorization: Bearer


async def _forward_mineru(request: Request, path: str):
    body = await request.body()
    if len(body) > _MAX_REQUEST_BYTES:
        return JSONResponse({"error": "请求体超过 10MB 上限"}, status_code=413)
    api_key = request.headers.get("X-API-Key")
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("x-api-key", None)
    headers["authorization"] = f"Bearer {api_key or ''}"
    url = f"https://mineru.net/api/v4{path}"
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=10.0)) as client:
            resp = await client.request(
                request.method,
                url,
                content=body,
                headers=headers,
            )
        return JSONResponse(
            status_code=resp.status_code,
            content=resp.json() if resp.content else {"error": "empty response"},
        )
    except httpx.HTTPError as e:
        return JSONResponse({"error": f"MinerU 代理请求失败: {e}"}, status_code=502)


@app.post("/proxy/mineru/extract/task")
async def proxy_mineru_create_task(request: Request):
    return await _forward_mineru(request, "/extract/task")


@app.get("/proxy/mineru/extract/task/{task_id}")
async def proxy_mineru_task_status(task_id: str, request: Request):
    return await _forward_mineru(request, f"/extract/task/{task_id}")


@app.get("/proxy/mineru/extract/result/{task_id}")
async def proxy_mineru_result(task_id: str, request: Request):
    return await _forward_mineru(request, f"/extract/result/{task_id}")


@app.get("/proxy/health")
async def proxy_health():
    return {"status": "ok"}
