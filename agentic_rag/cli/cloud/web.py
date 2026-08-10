"""DuckDuckGo/Tavily search and bounded public Web capture."""

from __future__ import annotations

import html as html_lib
import re
from urllib.parse import parse_qs, quote, unquote, urlsplit

from bs4 import BeautifulSoup

from ..cancellation import CancellationToken
from ..errors import ConfigurationError, UpstreamError
from ..models import CapturedPage, SearchResult
from ..security import validate_http_url
from .transport import HttpTransport


class WebClient:
    MAX_BYTES = 2 * 1024 * 1024
    MAX_REDIRECTS = 5

    def __init__(self, *, transport: HttpTransport | None = None) -> None:
        self.transport = transport or HttpTransport("Web", 30.0, 2)

    def search(self, query: str, provider: str, api_key: str, cancel: CancellationToken, limit: int = 6) -> list[SearchResult]:
        query, limit = query.strip(), max(1, min(int(limit), 10))
        if not query:
            raise ConfigurationError("Search query is required")
        if provider == "tavily":
            if not api_key:
                raise ConfigurationError("Tavily API key is not configured")
            data = self.transport.request_json("POST", "https://api.tavily.com/search", json_body={"api_key": api_key, "query": query, "max_results": limit, "search_depth": "basic"}, cancel=cancel)
            return [SearchResult(str(item.get("title", "")), str(item.get("url", "")), str(item.get("content", ""))[:500]) for item in (data.get("results") or [])[:limit]]
        with self.transport.stream("GET", f"https://html.duckduckgo.com/html/?q={quote(query)}", headers={"User-Agent": "AutoMemory/0.2"}, cancel=cancel) as response:
            raw = response.read()
        if len(raw) > self.MAX_BYTES:
            raise UpstreamError("DuckDuckGo response exceeded the allowed size")
        html = raw.decode(response.encoding or "utf-8", errors="replace")
        soup = BeautifulSoup(html, "html.parser")
        results = []
        for result in soup.select(".result"):
            anchor = result.select_one("a.result__a")
            snippet = result.select_one(".result__snippet")
            if not anchor:
                continue
            href = html_lib.unescape(str(anchor.get("href") or ""))
            target = unquote(parse_qs(urlsplit(href).query).get("uddg", [href])[0])
            try:
                validate_http_url(target)
            except Exception:
                continue
            results.append(SearchResult(anchor.get_text(" ", strip=True), target, snippet.get_text(" ", strip=True)[:500] if snippet else ""))
            if len(results) >= limit:
                break
        return results

    def fetch(self, url: str, cancel: CancellationToken) -> CapturedPage:
        current = validate_http_url(url)
        for _ in range(self.MAX_REDIRECTS + 1):
            cancel.checkpoint()
            with self.transport.stream("GET", current, headers={"User-Agent": "AutoMemory/0.2"}, cancel=cancel) as response:
                if response.is_redirect:
                    location = response.headers.get("location", "")
                    current = validate_http_url(str(response.url.join(location)))
                    continue
                content_type = response.headers.get("content-type", "").lower()
                if not any(item in content_type for item in ("text/html", "application/xhtml+xml", "text/plain")):
                    raise UpstreamError("Target is not readable HTML or text")
                declared = int(response.headers.get("content-length", "0") or 0)
                if declared > self.MAX_BYTES:
                    raise UpstreamError("Web page exceeded the 2MB limit")
                chunks, total = [], 0
                for chunk in response.iter_bytes():
                    cancel.checkpoint()
                    total += len(chunk)
                    if total > self.MAX_BYTES:
                        raise UpstreamError("Web page exceeded the 2MB limit")
                    chunks.append(chunk)
                raw = b"".join(chunks).decode(response.encoding or "utf-8", errors="replace")
                return self._extract(raw, str(response.url), content_type)
        raise UpstreamError("Web page exceeded the redirect limit")

    @staticmethod
    def _extract(raw: str, url: str, content_type: str) -> CapturedPage:
        soup = BeautifulSoup(raw, "html.parser")
        title = soup.title.get_text(" ", strip=True) if soup.title else url
        for node in soup.select("script,style,noscript,nav,footer,header,aside,iframe,svg"):
            node.decompose()
        text = "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())
        if not text:
            raise UpstreamError("No readable page text was extracted")
        return CapturedPage(title or url, url, text[:500_000], content_type)

    def close(self) -> None:
        self.transport.close()
