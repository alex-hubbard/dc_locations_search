from __future__ import annotations

import pandas as pd

from dc_locations_search import llm, persist, pipeline, search


def _install_mocks(monkeypatch, tavily_response, llm_response):
    """Mock the network boundary: search backend + LLM call + client factory."""
    monkeypatch.setattr(
        search, "_backend_search", lambda query, max_results: tavily_response["results"]
    )
    monkeypatch.setattr(pipeline, "_make_client", lambda: object())
    # llm.extract is imported lazily inside extract.process_one.
    monkeypatch.setattr(
        llm, "extract", lambda client, header, text, label, **kw: (llm_response, "mock-model")
    )


def test_pipeline_writes_wide_row_with_provenance(
    monkeypatch, isolated_data_dirs, sample_csv, tavily_response, llm_response
):
    _install_mocks(monkeypatch, tavily_response, llm_response)
    result = pipeline.run(sample_csv, limit=1)

    assert result["processed"] == 1
    assert result["failed"] == 0

    df = pd.read_csv(result["paths"]["wide_csv"])
    assert len(df) == 1
    row = df.iloc[0]
    assert row["facility_name"] == "QTS Atlanta-Metro"
    assert row["cooling_type"] == "air_cooled"
    assert row["capacity_mw"] == 72.0
    assert row["source"] == "dc_locations_search"
    # Provenance carried through.
    assert "datacenterdynamics.com" in str(row["cooling_type_source_url"])
    assert row["cooling_type_confidence"] == "high"


def test_inventory_subset_written(
    monkeypatch, isolated_data_dirs, sample_csv, tavily_response, llm_response
):
    _install_mocks(monkeypatch, tavily_response, llm_response)
    result = pipeline.run(sample_csv, limit=1)
    inv = pd.read_csv(result["paths"]["inventory_csv"])
    from dc_locations_search.schema import INVENTORY_COLUMNS

    assert list(inv.columns) == list(INVENTORY_COLUMNS)


def test_citations_sidecar_references_real_urls(
    monkeypatch, isolated_data_dirs, sample_csv, tavily_response, llm_response
):
    _install_mocks(monkeypatch, tavily_response, llm_response)
    result = pipeline.run(sample_csv, limit=1)
    cit = pd.read_csv(result["paths"]["citations_csv"])
    assert (cit["field"] == "cooling_type").any()
    assert cit["source_url"].str.contains("datacenterdynamics.com").any()


def test_reddit_filtered_from_sources(
    monkeypatch, isolated_data_dirs, sample_csv, tavily_response, llm_response
):
    _install_mocks(monkeypatch, tavily_response, llm_response)
    result = pipeline.run(sample_csv, limit=1)
    df = pd.read_csv(result["paths"]["wide_csv"])
    assert "reddit.com" not in str(df.iloc[0]["source_urls"])
    assert int(df.iloc[0]["n_articles_used"]) == 2


def test_resume_skips_completed(
    monkeypatch, isolated_data_dirs, sample_csv, tavily_response, llm_response
):
    _install_mocks(monkeypatch, tavily_response, llm_response)
    result1 = pipeline.run(sample_csv)
    assert result1["processed"] == 3
    status = persist.load_status_map()
    assert sum(v == "success" for v in status.values()) == 3

    # Second run with resume should process 0 (all DCs already succeeded).
    result2 = pipeline.run(sample_csv)
    assert result2["processed"] == 0


def test_dry_run_makes_no_llm_call(
    monkeypatch, isolated_data_dirs, sample_csv, tavily_response
):
    # Pre-seed the search cache so dry-run can emit a cache-only row.
    monkeypatch.setattr(
        search, "_backend_search", lambda query, max_results: tavily_response["results"]
    )
    df = pd.read_csv(sample_csv)
    from dc_locations_search.input_csv import compute_dc_id

    dc_id = compute_dc_id(df.iloc[0]["name"], df.iloc[0]["address"], df.iloc[0]["city"], df.iloc[0]["state"])
    search.gather_articles(dc_id, df.iloc[0]["name"], df.iloc[0]["city"], df.iloc[0]["state"])

    # If the LLM were called, this would raise.
    monkeypatch.setattr(
        llm, "extract", lambda *a, **k: (_ for _ in ()).throw(AssertionError("LLM called in dry run"))
    )
    result = pipeline.run(sample_csv, limit=1, dry_run=True)
    assert result["dry_run"] is True
