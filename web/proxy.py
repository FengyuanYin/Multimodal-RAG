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
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
import asyncio
import base64
import io
import ipaddress
import json
import mimetypes
import os
import socket
import zipfile
from pathlib import PurePosixPath
from urllib.parse import unquote
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
    allow_headers=["Content-Type", "X-API-Key", "X-File-Name"],
)


@app.middleware("http")
async def allow_local_network_preflight(request: Request, call_next):
    """允许受信任 Pages Origin 在用户授权后访问本机代理。"""
    response = await call_next(request)
    if request.headers.get("access-control-request-private-network", "").lower() == "true":
        response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response

_TIMEOUT = httpx.Timeout(120.0, connect=10.0)
_FETCH_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
_MAX_FETCH_BYTES = 2 * 1024 * 1024  # 网页抓取大小上限 2MB
_MAX_REQUEST_BYTES = 10 * 1024 * 1024
_MAX_STREAM_BYTES = 32 * 1024 * 1024
_MAX_MINERU_FILE_BYTES = 50 * 1024 * 1024
_MAX_MINERU_ARCHIVE_BYTES = 64 * 1024 * 1024
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
        if payload.get("stream") is True:
            client = httpx.AsyncClient(timeout=_TIMEOUT)
            try:
                upstream = await client.send(
                    client.build_request(
                        request.method,
                        url,
                        content=json.dumps(payload).encode("utf-8"),
                        headers=headers,
                    ),
                    stream=True,
                )
            except Exception:
                await client.aclose()
                raise
            if upstream.status_code >= 400:
                content = await upstream.aread()
                await upstream.aclose()
                await client.aclose()
                try:
                    detail = json.loads(content) if content else {"error": "empty response"}
                except Exception:
                    detail = {"error": "上游模型请求失败"}
                return JSONResponse(status_code=upstream.status_code, content=detail)

            async def relay():
                total = 0
                try:
                    async for chunk in upstream.aiter_bytes():
                        total += len(chunk)
                        if total > _MAX_STREAM_BYTES:
                            yield b'event: error\ndata: {"error":{"message":"stream response exceeded 32MB"}}\n\n'
                            break
                        yield chunk
                finally:
                    await upstream.aclose()
                    await client.aclose()

            return StreamingResponse(
                relay(),
                status_code=upstream.status_code,
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
            )

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


# ── MinerU 官方 API 适配器 ──
# GitHub Pages 不能直接调用 MinerU（官方端点无浏览器 CORS）。代理按官方流程执行：
# 申请预签名 URL → 上传文件 → 轮询 batch → 下载并规范化结果。

_MINERU_API_BASE = "https://mineru.net/api/v4"


def _mineru_message(payload: dict, fallback: str) -> str:
    message = str(payload.get("msg") or payload.get("message") or fallback)
    return message[:300]


async def _read_limited(response: httpx.Response, limit: int) -> bytes:
    chunks = []
    total = 0
    async for chunk in response.aiter_bytes():
        total += len(chunk)
        if total > limit:
            raise ValueError("MinerU 结果压缩包超过 64MB 上限")
        chunks.append(chunk)
    return b"".join(chunks)


def _extract_mineru_archive(archive: bytes) -> dict:
    """从官方 ZIP 中优先读取 content_list，保留页码、表格和受限图片。"""
    try:
        bundle = zipfile.ZipFile(io.BytesIO(archive))
    except zipfile.BadZipFile as exc:
        raise ValueError("MinerU 返回的结果不是有效 ZIP") from exc

    infos = [item for item in bundle.infolist() if not item.is_dir()]
    if sum(item.file_size for item in infos) > _MAX_MINERU_ARCHIVE_BYTES:
        raise ValueError("MinerU 解压后内容超过 64MB 上限")
    names = {item.filename: item for item in infos}
    content_name = next((name for name in names if name.endswith("_content_list.json")), None)
    pages = {}
    media = []
    image_total = 0

    if content_name:
        try:
            content_items = json.loads(bundle.read(content_name).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("MinerU content_list.json 无法解析") from exc
        image_seq = 0
        for item in content_items if isinstance(content_items, list) else []:
            page = max(1, int(item.get("page_idx", 0)) + 1)
            kind = str(item.get("type") or "text").lower()
            text = str(item.get("text") or item.get("table_body") or item.get("latex") or "").strip()
            captions = item.get("img_caption") or item.get("table_caption") or []
            if isinstance(captions, list):
                caption = " ".join(str(value) for value in captions if value)
            else:
                caption = str(captions or "")
            if kind == "table" and text:
                text = f"{caption}\n{text}".strip()
            elif caption and caption not in text:
                text = f"{caption}\n{text}".strip()
            if text:
                pages.setdefault(page, []).append(text)

            image_path = str(item.get("img_path") or "").replace("\\", "/")
            if kind == "image" and image_path:
                archive_path = str(PurePosixPath(content_name).parent / image_path)
                if archive_path not in names and image_path in names:
                    archive_path = image_path
                info = names.get(archive_path)
                if info and info.file_size <= 4 * 1024 * 1024 and image_total + info.file_size <= 12 * 1024 * 1024:
                    raw = bundle.read(archive_path)
                    mime = mimetypes.guess_type(archive_path)[0] or "image/png"
                    image_seq += 1
                    image_total += len(raw)
                    media.append({
                        "id": f"mineru_image_{image_seq}_p{page}",
                        "page": page,
                        "label": f"图{image_seq}",
                        "type": "image",
                        "num": image_seq,
                        "dataUrl": f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}",
                        "caption": caption,
                        "quality": "exact",
                        "extractionMethod": "mineru_official_v4",
                    })

    if not pages:
        markdown_names = [name for name in names if name.lower().endswith(".md")]
        if not markdown_names:
            raise ValueError("MinerU 结果中没有 Markdown 或 content_list.json")
        markdown_name = max(markdown_names, key=lambda name: names[name].file_size)
        markdown = bundle.read(markdown_name).decode("utf-8", errors="replace").strip()
        if not markdown:
            raise ValueError("MinerU 返回的 Markdown 为空")
        pages[1] = [markdown]

    return {
        "pages": [{"page": page, "text": "\n\n".join(parts)} for page, parts in sorted(pages.items())],
        "media": media,
    }


@app.post("/proxy/mineru/parse")
async def proxy_mineru_parse(request: Request):
    api_key = (request.headers.get("X-API-Key") or "").strip()
    if not api_key:
        return JSONResponse({"error": "缺少 MinerU API Key"}, status_code=401)
    body = await request.body()
    if not body:
        return JSONResponse({"error": "PDF 文件为空"}, status_code=400)
    if len(body) > _MAX_MINERU_FILE_BYTES:
        return JSONResponse({"error": "PDF 超过代理 50MB 上限"}, status_code=413)
    filename = unquote(request.headers.get("X-File-Name") or "document.pdf")
    filename = os.path.basename(filename.replace("\\", "/"))[:180]
    if not filename.lower().endswith(".pdf"):
        return JSONResponse({"error": "MinerU 解析仅接受 PDF 文件"}, status_code=415)

    auth = {"Authorization": f"Bearer {api_key}"}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
            create = await client.post(
                f"{_MINERU_API_BASE}/file-urls/batch",
                headers=auth,
                json={
                    "files": [{"name": filename, "data_id": os.urandom(12).hex()}],
                    "model_version": "vlm",
                    "enable_formula": True,
                    "enable_table": True,
                    "language": "ch",
                },
            )
            create_payload = create.json() if create.content else {}
            if create.status_code >= 400 or create_payload.get("code") not in (None, 0):
                return JSONResponse({"error": _mineru_message(create_payload, "MinerU 创建上传任务失败")}, status_code=502)
            data = create_payload.get("data") or {}
            batch_id = data.get("batch_id")
            upload_urls = data.get("file_urls") or []
            if not batch_id or not upload_urls:
                return JSONResponse({"error": "MinerU 未返回 batch_id 或上传地址"}, status_code=502)

            upload = await client.put(upload_urls[0], content=body)
            if upload.status_code >= 400:
                return JSONResponse({"error": f"上传 PDF 到 MinerU 存储失败（HTTP {upload.status_code}）"}, status_code=502)

            deadline = asyncio.get_running_loop().time() + 300
            result_item = None
            while asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(2)
                status = await client.get(f"{_MINERU_API_BASE}/extract-results/batch/{batch_id}", headers=auth)
                status_payload = status.json() if status.content else {}
                if status.status_code >= 400 or status_payload.get("code") not in (None, 0):
                    return JSONResponse({"error": _mineru_message(status_payload, "MinerU 查询解析状态失败")}, status_code=502)
                results = (status_payload.get("data") or {}).get("extract_result") or []
                result_item = results[0] if results else None
                state = str((result_item or {}).get("state") or "").lower()
                if state == "done":
                    break
                if state == "failed":
                    return JSONResponse({"error": str(result_item.get("err_msg") or "MinerU 解析失败")[:300]}, status_code=422)
            else:
                return JSONResponse({"error": "MinerU 解析超过 5 分钟，请稍后重试"}, status_code=504)

            archive_url = (result_item or {}).get("full_zip_url")
            if not archive_url:
                return JSONResponse({"error": "MinerU 已完成但未返回结果下载地址"}, status_code=502)
            async with client.stream("GET", archive_url) as archive_response:
                archive_response.raise_for_status()
                archive = await _read_limited(archive_response, _MAX_MINERU_ARCHIVE_BYTES)
        return JSONResponse(_extract_mineru_archive(archive))
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)
    except (httpx.HTTPError, zipfile.BadZipFile):
        return JSONResponse({"error": "MinerU 上游请求或结果下载失败"}, status_code=502)


@app.get("/proxy/health")
async def proxy_health():
    return {"status": "ok"}
