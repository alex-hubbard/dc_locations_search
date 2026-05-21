"""Persistence: resumable JSONL log, atomic writes, and output artifacts.

Outputs (to data/processed/, run-stamped):
- dc_attributes_<stamp>.{csv,parquet,xlsx}  — full wide superset table
- inventory_subset_<stamp>.csv               — exactly INVENTORY_COLUMNS
- citations_<stamp>.csv + sources_<stamp>.jsonl — per-value provenance sidecar
- dc_attributes_latest.parquet               — stable non-stamped path
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from dc_locations_search import config
from dc_locations_search.schema import (
    EXTRACTED_FIELDS,
    conform,
    empty_extraction_df,
    to_inventory_subset,
)

_LOG_LOCK = threading.Lock()


def _na_to_none(v: Any) -> Any:
    """Coerce pandas NA/NaN scalars to None so json.dumps doesn't choke."""
    if isinstance(v, list):
        return v
    try:
        return None if pd.isna(v) else v
    except (TypeError, ValueError):
        return v


# --- Atomic write ------------------------------------------------------------

def atomic_write_text(path: Path | str, text: str, encoding: str = "utf-8") -> None:
    """Write atomically via a same-dir .tmp + os.replace (from permit ocr.py)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w", encoding=encoding) as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


# --- Resumability log --------------------------------------------------------

def log_result(dc_id: str, status: str, **extra: Any) -> None:
    """Append one JSONL record keyed by dc_id (thread-safe)."""
    record = {
        "dc_id": dc_id,
        "status": status,
        "logged_at": datetime.now().isoformat(timespec="seconds"),
        **extra,
    }
    config.PROCESSING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _LOG_LOCK:
        with open(config.PROCESSING_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


def load_status_map() -> dict[str, str]:
    """Return {dc_id: status}, last write wins."""
    path = config.PROCESSING_LOG_PATH
    status: dict[str, str] = {}
    if not path.exists():
        return status
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                status[rec["dc_id"]] = rec["status"]
            except (json.JSONDecodeError, KeyError):
                continue
    return status


# --- Output writing ----------------------------------------------------------

def _stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _citations_long(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (dc_id, field, source_url, confidence) for populated fields."""
    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        for field in EXTRACTED_FIELDS:
            val = r.get(field)
            if val is None or (not isinstance(val, list) and pd.isna(val)):
                continue
            rows.append(
                {
                    "dc_id": r.get("dc_id"),
                    "facility_name": r.get("facility_name"),
                    "field": field,
                    "value": val,
                    "source_url": r.get(f"{field}_source_url"),
                    "confidence": r.get(f"{field}_confidence"),
                }
            )
    return pd.DataFrame(rows)


def rows_to_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a conformed wide DataFrame from accumulated row dicts."""
    if not rows:
        return empty_extraction_df()
    return conform(pd.DataFrame(rows))


def write_outputs(rows: list[dict[str, Any]], output_dir: Path | None = None) -> dict[str, Path]:
    """Write all output artifacts for the accumulated rows. Returns paths."""
    output_dir = Path(output_dir) if output_dir else config.PROCESSED_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    df = rows_to_df(rows)
    stamp = _stamp()

    paths: dict[str, Path] = {}

    wide_csv = output_dir / f"dc_attributes_{stamp}.csv"
    df.to_csv(wide_csv, index=False)
    paths["wide_csv"] = wide_csv

    try:
        wide_parquet = output_dir / f"dc_attributes_{stamp}.parquet"
        df.to_parquet(wide_parquet, index=False)
        paths["wide_parquet"] = wide_parquet
        # Stable latest pointer.
        latest = output_dir / "dc_attributes_latest.parquet"
        df.to_parquet(latest, index=False)
        paths["latest_parquet"] = latest
    except Exception as e:  # noqa: BLE001 - parquet engine optional
        logger.warning(f"Parquet write skipped: {e}")

    try:
        wide_xlsx = output_dir / f"dc_attributes_{stamp}.xlsx"
        df.to_excel(wide_xlsx, index=False)
        paths["wide_xlsx"] = wide_xlsx
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Excel write skipped: {e}")

    inv = to_inventory_subset(df)
    inv_csv = output_dir / f"inventory_subset_{stamp}.csv"
    inv.to_csv(inv_csv, index=False)
    paths["inventory_csv"] = inv_csv

    citations = _citations_long(df)
    cit_csv = output_dir / f"citations_{stamp}.csv"
    citations.to_csv(cit_csv, index=False)
    paths["citations_csv"] = cit_csv

    sources_jsonl = output_dir / f"sources_{stamp}.jsonl"
    lines = []
    for _, r in df.iterrows():
        lines.append(
            json.dumps(
                {
                    "dc_id": _na_to_none(r.get("dc_id")),
                    "facility_name": _na_to_none(r.get("facility_name")),
                    "source_urls": r.get("source_urls") if isinstance(r.get("source_urls"), list) else [],
                    "n_articles_used": int(r["n_articles_used"]) if pd.notna(r.get("n_articles_used")) else 0,
                    "model_used": _na_to_none(r.get("model_used")),
                }
            )
        )
    atomic_write_text(sources_jsonl, "\n".join(lines) + ("\n" if lines else ""))
    paths["sources_jsonl"] = sources_jsonl

    logger.info(f"Wrote {len(df)} rows to {output_dir}")
    return paths
