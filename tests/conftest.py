"""Shared pytest fixtures and path helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_csv() -> Path:
    return FIXTURES / "sample_input.csv"


@pytest.fixture
def tavily_response() -> dict:
    return json.loads((FIXTURES / "tavily_response.json").read_text())


@pytest.fixture
def llm_response() -> dict:
    return json.loads((FIXTURES / "llm_response.json").read_text())


@pytest.fixture
def isolated_data_dirs(tmp_path, monkeypatch):
    """Point config data paths at a tmp dir so tests don't touch real data/."""
    from dc_locations_search import config

    interim = tmp_path / "interim"
    processed = tmp_path / "processed"
    cache = interim / "search_cache"
    for d in (interim, processed, cache):
        d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "INTERIM_DATA_DIR", interim)
    monkeypatch.setattr(config, "PROCESSED_DATA_DIR", processed)
    monkeypatch.setattr(config, "SEARCH_CACHE_DIR", cache)
    monkeypatch.setattr(config, "PROCESSING_LOG_PATH", interim / "processing_log.jsonl")
    return tmp_path
