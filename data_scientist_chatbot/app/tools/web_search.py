"""Web search tool with multiple provider adapters."""

import os
import logging
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod

import httpx

logger = logging.getLogger(__name__)


class SearchAdapter(ABC):
    @abstractmethod
    async def search(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        pass


class DuckDuckGoAdapter(SearchAdapter):
    async def search(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        try:
            from ddgs import DDGS

            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=num_results))
                return [
                    {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
                    for r in results
                ]
        except ImportError:
            try:
                from duckduckgo_search import DDGS

                with DDGS() as ddgs:
                    results = list(ddgs.text(query, max_results=num_results))
                    return [
                        {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
                        for r in results
                    ]
            except ImportError:
                logger.warning("Neither ddgs nor duckduckgo-search installed")
                return [
                    {"title": "DuckDuckGo unavailable", "url": "", "snippet": "Install ddgs package: pip install ddgs"}
                ]
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return []


class BraveSearchAdapter(SearchAdapter):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.search.brave.com/res/v1/web/search"

    async def search(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        if not self.api_key:
            return [{"title": "Error", "url": "", "snippet": "Brave API key not configured"}]
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    self.base_url,
                    params={"q": query, "count": num_results},
                    headers={"X-Subscription-Token": self.api_key, "Accept": "application/json"},
                    timeout=10.0,
                )
                response.raise_for_status()
                data = response.json()
                results = data.get("web", {}).get("results", [])
                return [
                    {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("description", "")}
                    for r in results[:num_results]
                ]
        except Exception as e:
            logger.error(f"Brave search failed: {e}")
            return [{"title": "Error", "url": "", "snippet": f"Brave search failed: {e}"}]


class SearXNGAdapter(SearchAdapter):
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    async def search(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        if not self.base_url:
            return [{"title": "Error", "url": "", "snippet": "SearXNG URL not configured"}]
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/search", params={"q": query, "format": "json"}, timeout=10.0
                )
                response.raise_for_status()
                data = response.json()
                results = data.get("results", [])
                return [
                    {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
                    for r in results[:num_results]
                ]
        except Exception as e:
            logger.error(f"SearXNG search failed: {e}")
            return [{"title": "Error", "url": "", "snippet": f"SearXNG search failed: {e}"}]


def get_search_adapter(config: Dict[str, Any]) -> SearchAdapter:
    provider = config.get("provider", "duckduckgo")
    if provider == "brave":
        return BraveSearchAdapter(config.get("api_key", ""))
    elif provider == "searxng":
        return SearXNGAdapter(config.get("url", ""))
    return DuckDuckGoAdapter()


async def web_search(query: str, config: Dict[str, Any], num_results: int = 20) -> str:
    import json
    import builtins
    from datetime import datetime

    adapter = get_search_adapter(config)
    provider = config.get("provider", "duckduckgo")

    logger.info(f"[SEARCH_STATUS] searching|{query}|{provider}")

    results = await adapter.search(query, num_results)

    if not results:
        return "No search results found."

    logger.info(f"[SEARCH_STATUS] results|{len(results)}|{query}")

    for r in results[:3]:
        if r.get("url"):
            logger.info(f"[SEARCH_STATUS] browsing|{r['url'][:80]}")

    from urllib.parse import urlparse

    def get_domain(url):
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.replace("www.", "")
            return domain
        except:
            return ""

    output = ["## Web Search Results\n"]
    for i, r in enumerate(results, 1):
        url = r.get("url", "")
        domain = get_domain(url)
        # Use markdown hyperlink for the source
        source_link = f"[{domain}]({url})" if url else f"`{domain}`"
        output.append(f"{i}. **{r['title']}**\n   {r['snippet']}\n   {source_link}")

    # Build sources line with hyperlinks
    source_links = []
    seen_domains = set()
    for r in results:
        url = r.get("url", "")
        domain = get_domain(url)
        if domain and domain not in seen_domains and len(source_links) < 5:
            source_links.append(f"[{domain}]({url})")
            seen_domains.add(domain)

    sources_line = "  ".join(source_links)
    output.append(f"\n**Sources:** {sources_line}")

    # Sanitize results to prevent breaking the comment block
    for r in results:
        if "snippet" in r:
            r["snippet"] = r["snippet"].replace("-->", "--&gt;").replace("SEARCH_DATA_END", "SEARCH_DATA_END_ESCAPED")

    structured_data = {
        "query": query,
        "provider": provider,
        "timestamp": datetime.now().isoformat(),
        "result_count": len(results),
        "results": results,
        "sources": list(seen_domains),
    }
    import base64

    # Base64 encode to prevent any JSON parsing/regex issues in streaming service
    json_bytes = json.dumps(structured_data, ensure_ascii=False).encode("utf-8")
    b64_data = base64.b64encode(json_bytes).decode("utf-8")

    output.append(f"\n\n<!-- SEARCH_DATA_B64\n{b64_data}\nSEARCH_DATA_END -->")

    return "\n\n".join(output)
