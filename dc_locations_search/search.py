"""Gather reference articles about a data center from the web.

Default backend is Tavily (clean, LLM-ready extracted content). Serper.dev is
available behind ``SEARCH_BACKEND=serper``. Results are deduped by host+path,
filtered against a light deny-list, cached to ``data/interim/search_cache``, and
aggregated into a single labeled text block the LLM can attribute values to.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import requests
from loguru import logger

from dc_locations_search import config


@dataclass
class Article:
    """A single search result fed to the LLM."""

    url: str
    title: str
    content: str


@dataclass
class SearchResult:
    """All articles gathered for one data center, plus provenance metadata."""

    dc_id: str
    queries: list[str]
    articles: list[Article]
    retrieved_at: str

    @property
    def source_urls(self) -> list[str]:
        return [a.url for a in self.articles]

    @property
    def n_articles(self) -> int:
        return len(self.articles)


# --- Query construction ------------------------------------------------------

def _clean(v: object) -> str:
    return str(v).strip() if v is not None and str(v).strip().lower() != "nan" else ""


def build_queries(name: object, city: object, state: object) -> list[str]:
    """Primary + focused secondary queries from name/city/state."""
    name_s, city_s, state_s = _clean(name), _clean(city), _clean(state)
    loc = " ".join(p for p in (city_s, state_s) if p)
    primary = f'"{name_s}" data center {loc}'.strip()
    secondary = [
        f"{name_s} data center {loc} cooling power capacity MW".strip(),
        f"{name_s} data center {loc} construction status".strip(),
    ]
    # Dedupe while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for q in [primary, *secondary]:
        if q and q not in seen:
            seen.add(q)
            out.append(q)
    return out


# --- Result filtering --------------------------------------------------------

def _result_key(url: str) -> str:
    p = urlparse(url)
    return f"{p.netloc.lower().lstrip('www.')}{p.path.rstrip('/')}"


def _denied(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    host = host[4:] if host.startswith("www.") else host
    return any(host == d or host.endswith("." + d) for d in config.SEARCH_DENY_DOMAINS)


def _dedup_filter(raw_results: list[dict], max_results: int) -> list[Article]:
    seen: set[str] = set()
    articles: list[Article] = []
    for r in raw_results:
        url = (r.get("url") or "").strip()
        if not url or _denied(url):
            continue
        key = _result_key(url)
        if key in seen:
            continue
        seen.add(key)
        content = r.get("raw_content") or r.get("content") or ""
        content = content.strip()[: config.PER_ARTICLE_CHAR_BUDGET]
        if not content:
            continue
        articles.append(Article(url=url, title=(r.get("title") or "").strip(), content=content))
        if len(articles) >= max_results:
            break
    return articles


# --- Backends ----------------------------------------------------------------

def _tavily_search(query: str, max_results: int) -> list[dict]:
    from tavily import TavilyClient

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not set in the environment/.env")
    client = TavilyClient(api_key=api_key)
    resp = client.search(
        query=query,
        search_depth=config.TAVILY_SEARCH_DEPTH,
        max_results=max_results,
        include_raw_content=True,
    )
    return resp.get("results", []) or []


def _serper_search(query: str, max_results: int) -> list[dict]:
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        raise RuntimeError("SERPER_API_KEY is not set in the environment/.env")
    resp = requests.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        data=json.dumps({"q": query, "num": max_results}),
        timeout=30,
    )
    resp.raise_for_status()
    organic = resp.json().get("organic", []) or []
    # Serper returns snippet only (no full body). Map to the common shape.
    return [
        {"url": o.get("link"), "title": o.get("title"), "content": o.get("snippet")}
        for o in organic[:max_results]
    ]


def _backend_search(query: str, max_results: int) -> list[dict]:
    backend = config.SEARCH_BACKEND.lower()
    if backend == "tavily":
        return _tavily_search(query, max_results)
    if backend == "serper":
        return _serper_search(query, max_results)
    raise ValueError(f"Unknown SEARCH_BACKEND: {config.SEARCH_BACKEND}")


# --- Caching -----------------------------------------------------------------

def _cache_path(dc_id: str) -> Path:
    return config.SEARCH_CACHE_DIR / f"{dc_id}.json"


def load_cached(dc_id: str) -> SearchResult | None:
    path = _cache_path(dc_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return SearchResult(
            dc_id=data["dc_id"],
            queries=data.get("queries", []),
            articles=[Article(**a) for a in data.get("articles", [])],
            retrieved_at=data.get("retrieved_at", ""),
        )
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(f"Corrupt search cache for {dc_id}: {e}; ignoring.")
        return None


def _save_cache(result: SearchResult) -> None:
    config.SEARCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "dc_id": result.dc_id,
        "queries": result.queries,
        "retrieved_at": result.retrieved_at,
        "articles": [asdict(a) for a in result.articles],
    }
    _cache_path(result.dc_id).write_text(json.dumps(payload, indent=2))


# --- Public API --------------------------------------------------------------

def gather_articles(
    dc_id: str,
    name: object,
    city: object,
    state: object,
    *,
    max_results: int | None = None,
    use_cache: bool = True,
) -> SearchResult:
    """Search the web for articles about one data center.

    Issues the primary query first; only runs secondary queries if the primary
    returned fewer than ``SEARCH_MIN_PRIMARY_RESULTS`` usable articles. Caches
    the aggregated result to ``data/interim/search_cache/<dc_id>.json``.
    """
    if use_cache:
        cached = load_cached(dc_id)
        if cached is not None:
            logger.debug(f"Using cached search results for {dc_id} ({cached.n_articles} articles)")
            return cached

    max_results = max_results or config.TAVILY_MAX_RESULTS
    queries = build_queries(name, city, state)
    collected: list[dict] = []
    used_queries: list[str] = []

    for i, query in enumerate(queries):
        results = _backend_search(query, max_results)
        used_queries.append(query)
        collected.extend(results)
        # Stop early once the primary query already yielded enough.
        if i == 0:
            primary_usable = _dedup_filter(collected, max_results)
            if len(primary_usable) >= config.SEARCH_MIN_PRIMARY_RESULTS:
                break

    articles = _dedup_filter(collected, max_results)
    result = SearchResult(
        dc_id=dc_id,
        queries=used_queries,
        articles=articles,
        retrieved_at=date.today().isoformat(),
    )
    _save_cache(result)
    logger.info(f"Gathered {result.n_articles} article(s) for {dc_id}")
    return result


def build_context(result: SearchResult) -> str:
    """Concatenate articles into a single labeled block for the LLM.

    Each block is prefixed with a source marker so the model can attribute each
    extracted value to a specific URL. Truncated to the total context budget.
    """
    blocks: list[str] = []
    total = 0
    for idx, art in enumerate(result.articles, 1):
        header = f"[[SOURCE {idx}: {art.url} | {art.title}]]"
        body = art.content
        block = f"{header}\n{body}\n"
        if total + len(block) > config.TOTAL_CONTEXT_CHAR_BUDGET and blocks:
            break
        blocks.append(block)
        total += len(block)
    return "\n".join(blocks)
