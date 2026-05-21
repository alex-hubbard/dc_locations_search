"""End-to-end pipeline: load input -> concurrent per-DC extraction -> outputs.

Resumable: a JSONL log keyed by dc_id lets re-runs skip already-successful DCs.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import openai
import pandas as pd
from loguru import logger
from tqdm import tqdm

from dc_locations_search import config, persist
from dc_locations_search.extract import process_one
from dc_locations_search.input_csv import ColumnMapping, load_input


def run(
    input_path: str | Path,
    *,
    mapping: ColumnMapping | None = None,
    output_dir: Path | None = None,
    limit: Optional[int] = None,
    max_workers: Optional[int] = None,
    resume: bool = True,
    save_every: Optional[int] = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run extraction over an input CSV.

    In dry_run mode no paid API is called: the input is validated, the queries
    that would run are printed, and (if cached search results exist) rows are
    emitted from cache without an LLM call.
    """
    config.ensure_dirs()
    df = load_input(input_path, mapping)
    logger.info(f"Loaded {len(df)} data center(s) from {input_path}")

    if resume and not dry_run:
        done = {k for k, v in persist.load_status_map().items() if v == "success"}
        before = len(df)
        df = df[~df["dc_id"].isin(done)].reset_index(drop=True)
        skipped = before - len(df)
        if skipped:
            logger.info(f"Resuming: skipping {skipped} already-successful DC(s).")

    if limit is not None:
        df = df.head(limit).reset_index(drop=True)

    if dry_run:
        return _dry_run(df, output_dir)

    if df.empty:
        logger.info("Nothing to process.")
        return {"processed": 0, "failed": 0, "paths": {}}

    client = _make_client()
    max_workers = max_workers or config.LLM_MAX_WORKERS
    save_every = save_every or config.SAVE_EVERY_N

    rows: list[dict[str, Any]] = []
    failed = 0
    last_paths: dict[str, Path] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_one, client, row): row["dc_id"]
            for _, row in df.iterrows()
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="extracting"):
            dc_id = futures[fut]
            try:
                row = fut.result()
                rows.append(row)
                persist.log_result(
                    dc_id, "success",
                    n_articles=row.get("n_articles_used"),
                    model_used=row.get("model_used"),
                )
            except Exception as e:  # noqa: BLE001
                failed += 1
                logger.error(f"Failed {dc_id}: {e}")
                persist.log_result(dc_id, "failed", error=str(e))
            if len(rows) and len(rows) % save_every == 0:
                last_paths = persist.write_outputs(rows, output_dir)

    last_paths = persist.write_outputs(rows, output_dir)
    logger.info(f"Done: {len(rows)} succeeded, {failed} failed.")
    return {"processed": len(rows), "failed": failed, "paths": last_paths}


def _make_client() -> openai.OpenAI:
    from dc_locations_search.llm import configure_llm

    return configure_llm()


def _dry_run(df: pd.DataFrame, output_dir: Path | None) -> dict[str, Any]:
    from dc_locations_search import search

    logger.info("DRY RUN — no paid API calls will be made.")
    logger.info(
        f"Config: backend={config.SEARCH_BACKEND}, model={config.LLM_MODEL}, "
        f"max_results={config.TAVILY_MAX_RESULTS}, workers={config.LLM_MAX_WORKERS}"
    )
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        queries = search.build_queries(row.get("facility_name"), row.get("city"), row.get("state"))
        logger.info(f"[{row['dc_id']}] {row.get('facility_name')} -> queries:")
        for q in queries:
            logger.info(f"    - {q}")
        # If a cached search exists, emit a cache-only row (still no API call).
        if search.load_cached(row["dc_id"]) is not None:
            rows.append(
                process_one(None, row, use_cache=True, llm_from_cache_only=True)
            )
    paths = persist.write_outputs(rows, output_dir) if rows else {}
    return {"processed": len(rows), "failed": 0, "dry_run": True, "paths": paths}
