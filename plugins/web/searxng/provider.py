"""SearXNG search — plugin form.

Subclasses :class:`agent.web_search_provider.WebSearchProvider`. Same JSON
API call (``/search?format=json``), same result normalization. The legacy
in-tree module ``tools.web_providers.searxng`` was removed in the same
commit that moved this code under ``plugins/``; this file is now the
canonical implementation.

Search-only — SearXNG aggregates results from upstream engines but does not
fetch/extract arbitrary URLs. ``supports_extract()`` returns False.

Config keys this provider responds to::

    web:
      search_backend: "searxng"     # explicit per-capability
      backend: "searxng"            # shared fallback

Env var::

    SEARXNG_URL=http://localhost:8080

GXTD-341 / GXTD-400: news-category routing + structured citations/favicons.
News queries are routed through SearXNG's ``categories=news`` first, with
``time_range=day`` for time-sensitive terms. Results include source domain,
citation strings, and Google favicon URLs for downstream rendering.
"""

from __future__ import annotations

import logging
import os
import urllib.parse
from typing import Any, Dict

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

_NEWS_QUERY_TERMS = (
    "news",
    "headline",
    "headlines",
    "current event",
    "current events",
    "today",
    "latest",
    "breaking",
    "what happened",
    "happening",
)

_FRESH_NEWS_QUERY_TERMS = (
    "today",
    "latest",
    "breaking",
    "headline",
    "headlines",
)


def _is_news_query(query: str) -> bool:
    normalized = " ".join((query or "").lower().split())
    return any(term in normalized for term in _NEWS_QUERY_TERMS)


def _is_fresh_news_query(query: str) -> bool:
    normalized = " ".join((query or "").lower().split())
    return any(term in normalized for term in _FRESH_NEWS_QUERY_TERMS)


def _result_source(url: str) -> str:
    parsed = urllib.parse.urlparse(url or "")
    return (parsed.netloc or parsed.path).removeprefix("www.")


def _google_favicon_url(url: str) -> str:
    source = _result_source(url)
    if not source:
        return ""
    return f"https://www.google.com/s2/favicons?domain={source}&sz=32"


def _searxng_url() -> str:
    """Return SEARXNG_URL from Hermes config-aware env, falling back to process env."""
    try:
        from hermes_cli.config import get_env_value

        val = get_env_value("SEARXNG_URL")
    except Exception:
        val = None
    if val is None:
        val = os.getenv("SEARXNG_URL", "")
    return (val or "").strip()


class SearXNGWebSearchProvider(WebSearchProvider):
    """Search via a user-hosted SearXNG instance."""

    @property
    def name(self) -> str:
        return "searxng"

    @property
    def display_name(self) -> str:
        return "SearXNG"

    def is_available(self) -> bool:
        """Return True when ``SEARXNG_URL`` is set."""
        return bool(_searxng_url())

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a search against the configured SearXNG instance."""
        import httpx

        base_url = _searxng_url().rstrip("/")
        if not base_url:
            return {"success": False, "error": "SEARXNG_URL is not set"}

        params: Dict[str, Any] = {
            "q": query,
            "format": "json",
            "pageno": 1,
        }

        def request(search_params: Dict[str, Any]) -> Dict[str, Any]:
            resp = httpx.get(
                f"{base_url}/search",
                params=search_params,
                timeout=15,
                headers={"Accept": "application/json"},
            )
            resp.raise_for_status()
            return resp.json()

        category_used = "general"
        try:
            if _is_news_query(query):
                news_params = dict(params)
                news_params["categories"] = "news"
                if _is_fresh_news_query(query):
                    news_params["time_range"] = "day"
                data = request(news_params)
                raw_results = data.get("results", [])
                category_used = "news_day" if news_params.get("time_range") == "day" else "news"
                if raw_results:
                    logger.info("SearXNG search '%s': used news category", query)
            else:
                data = request(params)
                raw_results = data.get("results", [])
        except httpx.HTTPStatusError as exc:
            logger.warning("SearXNG HTTP error: %s", exc)
            return {
                "success": False,
                "error": f"SearXNG returned HTTP {exc.response.status_code}",
            }
        except httpx.RequestError as exc:
            logger.warning("SearXNG request error: %s", exc)
            return {
                "success": False,
                "error": f"Could not reach SearXNG at {base_url}: {exc}",
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("SearXNG response parse error: %s", exc)
            return {
                "success": False,
                "error": "Could not parse SearXNG response as JSON",
            }

        if not raw_results and category_used.startswith("news"):
            try:
                data = request(params)
                raw_results = data.get("results", [])
                category_used = "general_fallback"
                if raw_results:
                    logger.info("SearXNG search '%s': fell back to general category", query)
            except Exception as exc:  # noqa: BLE001
                logger.warning("SearXNG general fallback after news-category search failed: %s", exc)

        # SearXNG may return a score field; sort descending and cap to limit.
        sorted_results = sorted(
            raw_results,
            key=lambda r: float(r.get("score", 0)),
            reverse=True,
        )[:limit]

        web_results = []
        for i, r in enumerate(sorted_results):
            title = str(r.get("title", ""))
            url = str(r.get("url", ""))
            source = _result_source(url)
            position = i + 1
            citation = f"[{position}] {title} — {url}" if title and url else url
            favicon_url = _google_favicon_url(url)
            item = {
                "title": title,
                "url": url,
                "description": str(r.get("content", "")),
                "position": position,
                "source": source,
                "citation": citation,
            }
            if favicon_url:
                item["favicon_url"] = favicon_url
                item["favicon_provider"] = "google"
            web_results.append(item)

        citations = []
        for i, item in enumerate(web_results, start=1):
            if item.get("url"):
                citations.append(
                    {
                        "position": i,
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "source": item.get("source", ""),
                        "favicon_url": item.get("favicon_url", ""),
                        "favicon_provider": item.get("favicon_provider", ""),
                    }
                )

        logger.info(
            "SearXNG search '%s': %d results (from %d raw, limit %d)",
            query,
            len(web_results),
            len(raw_results),
            limit,
        )

        rendered_sources = "\n".join(
            f"{entry['position']}. [{entry['source']}] {entry['title']} — {entry['url']} (favicon: {entry['favicon_url']})"
            for entry in citations
        )
        source_block = (
            "Use these sources in the final answer. Include citations like [1], [2], and include a Sources block.\n"
            f"Search category: {category_used}\n"
            f"Sources:\n{rendered_sources}"
            if citations
            else f"Search category: {category_used}\nSources: none"
        )

        return {
            "success": True,
            "data": {
                "web": web_results,
                "sources": citations,
                "citations": citations,
                "source_block": source_block,
                "search_category": category_used,
                "display": {
                    "requires_citations": True,
                    "favicon_provider": "google",
                    "source_block": source_block,
                },
            },
        }

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "SearXNG",
            "badge": "free · self-hosted",
            "tag": "Free, privacy-respecting metasearch. Point SEARXNG_URL at your instance.",
            "env_vars": [
                {
                    "key": "SEARXNG_URL",
                    "prompt": "SearXNG instance URL (e.g. http://localhost:8080)",
                    "url": "https://searx.space/",
                },
            ],
        }
