"""Hardened Web search and readable-page capture."""

from __future__ import annotations

import html as html_lib
import re
from urllib.parse import parse_qs, quote, unquote, urlsplit

import httpx

from ..events import CancelToken, EventCallback, JobProgress
from ..models import CapturedPage, SearchResult
from ..security import validate_public_url


class WebCaptureService:
    MAX_BYTES = 2 * 1024 * 1024
    MAX_REDIRECTS = 5
    TIMEOUT = httpx.Timeout(connect=10.0, read=30.0, write=30.0, pool=10.0)

    async def search(self, query: str, provider: str = "duckduckgo", api_key: str = "", limit: int = 6, cancel: CancelToken | None = None) -> list[SearchResult]:
        query = query.strip()
        if not query:
            raise ValueError("search query is required")
        if len(query) > 500:
            raise ValueError("search query exceeds 500 characters")
        if cancel:
            cancel.checkpoint()
        limit = max(1, min(int(limit), 10))
        if provider == "tavily":
            if not api_key:
                raise ValueError("Tavily API key is required")
            async with httpx.AsyncClient(timeout=self.TIMEOUT) as client:
                response = await client.post("https://api.tavily.com/search", json={"api_key": api_key, "query": query, "max_results": limit, "search_depth": "basic"})
                response.raise_for_status()
                data = response.json()
            return [SearchResult(str(item.get("title", "")), str(item.get("url", "")), str(item.get("content", ""))[:400]) for item in data.get("results", [])[:limit]]
        async with httpx.AsyncClient(timeout=self.TIMEOUT, follow_redirects=True) as client:
            response = await client.get(f"https://html.duckduckgo.com/html/?q={quote(query)}", headers={"User-Agent": "AutoMemory/0.1 (+local RAG client)"})
            response.raise_for_status()
        results = []
        pattern = re.compile(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>.*?<a[^>]+class="result__snippet"[^>]*>(.*?)</a>', re.I | re.S)
        for match in pattern.finditer(response.text):
            href = html_lib.unescape(match.group(1))
            query_values = parse_qs(urlsplit(href).query)
            target = unquote(query_values.get("uddg", [href])[0])
            try:
                validate_public_url(target)
            except ValueError:
                continue
            strip_tags = lambda value: html_lib.unescape(re.sub(r"<[^>]+>", "", value)).strip()
            results.append(SearchResult(strip_tags(match.group(2)), target, strip_tags(match.group(3))))
            if len(results) >= limit:
                break
        return results

    async def fetch(self, url: str, cancel: CancelToken | None = None) -> CapturedPage:
        current = validate_public_url(url)
        async with httpx.AsyncClient(timeout=self.TIMEOUT, follow_redirects=False) as client:
            for _ in range(self.MAX_REDIRECTS + 1):
                if cancel:
                    cancel.checkpoint()
                async with client.stream("GET", current, headers={"User-Agent": "AutoMemory/0.1"}) as response:
                    if response.is_redirect:
                        location = response.headers.get("location", "")
                        current = validate_public_url(str(response.url.join(location)))
                        continue
                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "").lower()
                    if not any(kind in content_type for kind in ("text/html", "application/xhtml+xml", "text/plain")):
                        raise ValueError("target is not readable HTML or text")
                    declared = int(response.headers.get("content-length", "0") or 0)
                    if declared > self.MAX_BYTES:
                        raise ValueError("page exceeds the 2MB limit")
                    body, total = [], 0
                    async for chunk in response.aiter_bytes():
                        if cancel:
                            cancel.checkpoint()
                        total += len(chunk)
                        if total > self.MAX_BYTES:
                            raise ValueError("page exceeds the 2MB limit")
                        body.append(chunk)
                    raw = b"".join(body).decode(response.encoding or "utf-8", errors="replace")
                    return self._extract(raw, str(response.url), content_type)
        raise ValueError("too many redirects")

    @staticmethod
    def _extract(raw: str, url: str, content_type: str) -> CapturedPage:
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(raw, "html.parser")
            title = soup.title.get_text(" ", strip=True) if soup.title else url
            for node in soup.select("script,style,noscript,nav,footer,header,aside,iframe,svg"):
                node.decompose()
            text = "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())
        except ImportError:
            title_match = re.search(r"<title[^>]*>(.*?)</title>", raw, re.I | re.S)
            title = html_lib.unescape(re.sub(r"<[^>]+>", "", title_match.group(1))).strip() if title_match else url
            text = html_lib.unescape(re.sub(r"<[^>]+>", " ", raw))
            text = re.sub(r"\s+", " ", text).strip()
        if not text:
            raise ValueError("no readable page text was extracted")
        return CapturedPage(title=title or url, url=url, text=text[:500_000], content_type=content_type)

    async def fetch_many(self, urls: list[str], emit: EventCallback | None = None, cancel: CancelToken | None = None, job_id: str = "web") -> list[CapturedPage | Exception]:
        output = []
        for index, url in enumerate(urls, 1):
            if cancel:
                cancel.checkpoint()
            if emit:
                emit(JobProgress(job_id, "web", "fetch", url, index - 1, len(urls)))
            try:
                output.append(await self.fetch(url, cancel))
            except Exception as exc:
                output.append(exc)
        return output
