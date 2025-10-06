"""Web search tool using DuckDuckGo"""

from typing import Optional


def search_web(query: str, max_results: int = 5) -> str:
    """Search the web for external context using DuckDuckGo"""
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        if not results:
            return f"No results found for: {query}"

        formatted_results = []
        for idx, result in enumerate(results, 1):
            title = result.get('title', 'No title')
            snippet = result.get('body', 'No description')
            url = result.get('href', '')
            formatted_results.append(f"{idx}. {title}\n{snippet}\nSource: {url}")

        return "\n\n".join(formatted_results)

    except Exception as e:
        return f"Web search failed: {str(e)}"
