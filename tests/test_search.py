from __future__ import annotations

from dc_locations_search import search
from dc_locations_search.search import Article, SearchResult, build_context, build_queries


def test_build_queries_primary_first():
    qs = build_queries("QTS Atlanta-Metro", "Atlanta", "GA")
    assert qs[0] == '"QTS Atlanta-Metro" data center Atlanta GA'
    assert len(qs) == 3


def test_build_queries_dedup_and_no_blank_loc():
    qs = build_queries("Foo DC", None, None)
    assert all(q.strip() for q in qs)
    assert len(qs) == len(set(qs))


def test_dedup_filter_removes_denied_and_dupes():
    raw = [
        {"url": "https://www.datacenterdynamics.com/a", "title": "A", "raw_content": "body a"},
        {"url": "https://datacenterdynamics.com/a/", "title": "A dup", "raw_content": "dup"},
        {"url": "https://www.reddit.com/r/x", "title": "R", "raw_content": "forum"},
        {"url": "https://example.com/b", "title": "B", "raw_content": "body b"},
    ]
    arts = search._dedup_filter(raw, max_results=10)
    urls = [a.url for a in arts]
    assert "https://www.reddit.com/r/x" not in urls
    # The www/non-www + trailing slash duplicate collapses to one.
    assert sum("datacenterdynamics.com" in u for u in urls) == 1
    assert len(arts) == 2


def test_dedup_filter_skips_empty_content():
    raw = [{"url": "https://example.com/x", "title": "X", "raw_content": "   "}]
    assert search._dedup_filter(raw, 10) == []


def test_build_context_labels_sources():
    sr = SearchResult(
        dc_id="abc",
        queries=["q"],
        articles=[Article(url="http://x", title="T", content="hello world")],
        retrieved_at="2026-05-21",
    )
    ctx = build_context(sr)
    assert "[[SOURCE 1: http://x | T]]" in ctx
    assert "hello world" in ctx


def test_cache_roundtrip(isolated_data_dirs):
    sr = SearchResult(
        dc_id="cacheid",
        queries=["q1"],
        articles=[Article(url="http://x", title="T", content="body")],
        retrieved_at="2026-05-21",
    )
    search._save_cache(sr)
    loaded = search.load_cached("cacheid")
    assert loaded is not None
    assert loaded.n_articles == 1
    assert loaded.articles[0].url == "http://x"
