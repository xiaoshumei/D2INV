"""
Web Search Tool for D2INV Agent — Phase 2.

Provides web search capability via a pluggable backend.
Default: DuckDuckGo HTML (no API key required).
Optionally: configurable base URL for custom/enterprise search backends.

Requires: requests, beautifulsoup4 (both already in requirements.txt)
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup

from agent.tools.base import BaseTool, ToolParam
from agent.session import Session


# ---------------------------------------------------------------------------
# Search backend abstraction
# ---------------------------------------------------------------------------

class SearchBackend:
    """Base search backend. Subclasses implement ``search(query, limit)``."""

    def search(self, query: str, limit: int = 10) -> List[Dict[str, str]]:
        raise NotImplementedError


class DuckDuckGoHTMLBackend(SearchBackend):
    """
    DuckDuckGo HTML search (lite-ish). No API key needed.

    Uses the html.duckduckgo.com/html/ endpoint and parses result cards.
    """

    BASE_URL = "https://html.duckduckgo.com/html/"

    def __init__(self, timeout: int = 10, max_retries: int = 2):
        self.timeout = timeout
        self.max_retries = max_retries

    def search(self, query: str, limit: int = 10) -> List[Dict[str, str]]:
        results = []
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(
                    self.BASE_URL,
                    data={"q": query},
                    timeout=self.timeout,
                    headers={"User-Agent": "Mozilla/5.0 (D2INV Agent)"},
                )
                if resp.status_code != 200:
                    raise RuntimeError(f"HTTP {resp.status_code}")

                soup = BeautifulSoup(resp.text, "html.parser")
                for result in soup.select(".result"):
                    title_el = result.select_one(".result__title")
                    link_el = result.select_one(".result__url")
                    snippet_el = result.select_one(".result__snippet")

                    if not title_el:
                        continue

                    title = title_el.get_text(strip=True)
                    link = link_el.get_text(strip=True) if link_el else ""
                    snippet = snippet_el.get_text(strip=True) if snippet_el else ""

                    # DuckDuckGo wraps links in redirect; extract the real URL
                    anchor = title_el.select_one("a") if hasattr(title_el, "select_one") else None
                    if anchor and anchor.get("href"):
                        href = anchor["href"]
                        import urllib.parse
                        parsed = urllib.parse.urlparse(href)
                        query_params = urllib.parse.parse_qs(parsed.query)
                        if "uddg" in query_params:
                            link = query_params["uddg"][0]

                    results.append({"title": title, "link": link, "snippet": snippet})
                    if len(results) >= limit:
                        break

                if results:
                    break  # success
            except Exception as exc:
                print(f"[WebSearch] attempt {attempt+1} failed: {exc}")
                if attempt < self.max_retries:
                    time.sleep(1 * (attempt + 1))  # backoff
        return results


# ---------------------------------------------------------------------------
# Web Search Tool
# ---------------------------------------------------------------------------

class WebSearchTool(BaseTool):
    """
    Search the web and return a list of results (title, link, snippet).

    Uses a backend abstraction so enterprise/custom search can be swapped in.
    """

    def __init__(self, session: Session, backend: Optional[SearchBackend] = None):
        super().__init__()
        self._session = session
        self._backend = backend or DuckDuckGoHTMLBackend()

    # ── BaseTool interface ────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Search the web for information. Returns a list of results with "
            "title, URL, and a text snippet. Useful for looking up facts, "
            "news, definitions, and external data sources. Use this when the "
            "user asks a question that requires current or external knowledge."
        )

    @property
    def parameters(self):
        return [
            ToolParam(
                name="query",
                type="string",
                description="Search query string.",
                required=True,
            ),
            ToolParam(
                name="limit",
                type="number",
                description="Maximum number of results to return (default 10, max 20).",
                required=False,
                default=10,
            ),
        ]

    @property
    def category(self) -> str:
        return "external"

    def _execute_impl(self, query: str, limit: int = 10) -> dict:
        limit = max(1, min(limit, 20))  # clamp

        # Rate-limit guard (simple per-session in-memory)
        now = time.time()
        last_key = "__web_search_last_time"
        min_interval = 2.0  # seconds between calls
        last_time = self._session.get_state(last_key, 0)
        if now - last_time < min_interval:
            wait = min_interval - (now - last_time)
            time.sleep(wait)
        self._session.set_state(last_key, time.time())

        results = self._backend.search(query, limit=limit)

        # Cache results in session for later agent use
        cache_key = f"__web_search_results_{hash(query) & 0xFFFFFFFF}"
        self._session.set_state(cache_key, results)

        return {
            "query": query,
            "count": len(results),
            "results": results[:limit],
        }


# ---------------------------------------------------------------------------
# Simple helper to create backend from config dict
# ---------------------------------------------------------------------------

def create_backend_from_config(config: Optional[Dict[str, Any]] = None) -> SearchBackend:
    """Factory to build a search backend from a config dict (future-proof)."""
    config = config or {}
    backend_type = config.get("type", "duckduckgo_html")
    if backend_type == "duckduckgo_html":
        return DuckDuckGoHTMLBackend(
            timeout=config.get("timeout", 10),
            max_retries=config.get("max_retries", 2),
        )
    # Future: add "google_custom_search", "bing", "elasticsearch", etc.
    raise ValueError(f"Unknown search backend type: {backend_type}")