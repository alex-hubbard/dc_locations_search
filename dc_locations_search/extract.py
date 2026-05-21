"""Per-data-center orchestration: search -> aggregate -> LLM -> flatten row."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import openai
import pandas as pd
from loguru import logger

from dc_locations_search import config, search
from dc_locations_search.schema import flatten_extraction


def _dc_header(row: pd.Series) -> str:
    parts = [
        f"Name: {row.get('facility_name')}",
        f"Address: {row.get('address')}",
        f"City: {row.get('city')}",
        f"State: {row.get('state')}",
    ]
    return "\n".join(p for p in parts if p and "None" not in p and "nan" not in p.lower())


def process_one(
    client: Optional[openai.OpenAI],
    row: pd.Series,
    *,
    use_cache: bool = True,
    llm_from_cache_only: bool = False,
) -> dict[str, Any]:
    """Search + extract one DC. Returns a flat row dict (conform() applied later).

    Raises on hard failure so the pipeline can mark the DC failed and resume later.
    """
    dc_id = row["dc_id"]
    sr = search.gather_articles(
        dc_id, row.get("facility_name"), row.get("city"), row.get("state"),
        use_cache=use_cache,
    )

    base: dict[str, Any] = {
        "dc_id": dc_id,
        "facility_name": row.get("facility_name"),
        "address": row.get("address"),
        "city": row.get("city"),
        "state": row.get("state"),
        "zip_code": row.get("zip_code"),
        "country": row.get("country"),
        "operator": row.get("operator"),
        "source": config.SOURCE_ID,
        "source_record_id": dc_id,
        "source_urls": sr.source_urls,
        "n_articles_used": sr.n_articles,
        "retrieved_at": sr.retrieved_at,
    }

    if sr.n_articles == 0:
        logger.warning(f"No articles found for {dc_id}; emitting identity-only row.")
        base["extraction_notes"] = "no articles found"
        base["model_used"] = None
        base["extracted_at"] = datetime.now().isoformat(timespec="seconds")
        return base

    if client is None and llm_from_cache_only:
        # Dry-run with cache: skip the LLM call.
        base["extraction_notes"] = "dry-run: LLM skipped"
        base["model_used"] = None
        base["extracted_at"] = datetime.now().isoformat(timespec="seconds")
        return base

    from dc_locations_search.llm import extract as llm_extract

    article_text = search.build_context(sr)
    parsed, model_used = llm_extract(client, _dc_header(row), article_text, label=dc_id)
    base["model_used"] = model_used
    base["extracted_at"] = datetime.now().isoformat(timespec="seconds")

    if parsed is None:
        raise RuntimeError(f"LLM extraction returned no data for {dc_id}")

    # Operator from the input wins only if the LLM didn't extract one.
    flat = flatten_extraction(parsed)
    extracted_operator = flat.get("operator")
    operator_missing = extracted_operator is None or (
        not isinstance(extracted_operator, list) and pd.isna(extracted_operator)
    )
    if operator_missing and base.get("operator"):
        flat.pop("operator", None)  # keep input operator
        flat.pop("operator_source_url", None)
        flat.pop("operator_confidence", None)
    base.update(flat)
    base["overall_confidence"] = parsed.get("overall_confidence")
    notes = parsed.get("extraction_notes")
    if notes:
        base["extraction_notes"] = notes
    return base
