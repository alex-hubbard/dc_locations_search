"""Extraction data model: a provenance-tracked superset of the canonical
lbl-data-center-map inventory schema.

One row per data center. Every *extracted* field carries two companion columns:
``<field>_source_url`` (the URL the value came from) and ``<field>_confidence``
(``high|medium|low``). A null value means "not stated in the gathered articles".

Canonical enums and helpers are imported directly from
``lbl_data_center_map.public.schema`` (an editable/path dependency) so there is a
single source of truth and no drift. The ``segment`` vocabulary comes from the
data-center-impact-model (``SEGMENT_CONFIG["order_lc"]``).
"""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

# Canonical inventory schema (single source of truth).
from lbl_data_center_map.public.schema import (  # noqa: F401
    CAPACITY_BASIS_VALUES,
    DC_TYPE_VALUES,
    INVENTORY_COLUMNS,
    STATUS_VALUES,
    conform as inventory_conform,
    normalize_address,
    normalize_facility_name,
)

# --- Local enum vocabularies -------------------------------------------------

CONFIDENCE_VALUES: frozenset[str] = frozenset({"high", "medium", "low"})

COOLING_TYPE_VALUES: frozenset[str] = frozenset(
    {"air_cooled", "liquid_cooled", "immersion", "hybrid", "unknown"}
)
PUE_BASIS_VALUES: frozenset[str] = frozenset(
    {"design", "reported", "target", "unknown"}
)
SQFT_BASIS_VALUES: frozenset[str] = frozenset(
    {"gross", "raised_floor", "white_space", "unknown"}
)
REDUNDANCY_TIER_VALUES: frozenset[str] = frozenset(
    {"tier_i", "tier_ii", "tier_iii", "tier_iv", "unknown"}
)

# data-center-impact-model SEGMENT_CONFIG["order_lc"]. Verbatim strings so the
# output joins to the model's location_data_segment_map.csv without translation.
SEGMENT_VALUES: tuple[str, ...] = (
    "Liquid Cooled AI",
    "Internet Giants",
    "Colocation - Internet Giants",
    "Colocation/Hosting - Enterprise",
    "Internal",
    "Comms SPs",
    "Enterprise Branch",
    "SMB",
    "Commercial Edge",
    "Telco Edge",
)

# Per-field enum constraint registry (used by conform()).
ENUM_FIELDS: dict[str, frozenset[str]] = {
    "status": STATUS_VALUES,
    "capacity_basis": CAPACITY_BASIS_VALUES,
    "dc_type": DC_TYPE_VALUES,
    "cooling_type": COOLING_TYPE_VALUES,
    "pue_basis": PUE_BASIS_VALUES,
    "square_footage_basis": SQFT_BASIS_VALUES,
    "redundancy_tier": REDUNDANCY_TIER_VALUES,
    "segment": frozenset(SEGMENT_VALUES),
}

# --- Field groups ------------------------------------------------------------

# Carried from the input CSV (not LLM-extracted).
IDENTITY_FIELDS: tuple[str, ...] = (
    "facility_name",
    "address",
    "city",
    "state",
    "zip_code",
    "country",
)

# Extracted by the LLM. Each gets <field>, <field>_source_url, <field>_confidence.
EXTRACTED_FIELDS: tuple[str, ...] = (
    # Identity / ownership
    "operator",
    "company",
    # Status & construction
    "status",
    "construction_phase",
    "year_in_service",
    "announced_date",
    "expected_completion",
    # Capacity & size
    "capacity_mw",
    "capacity_basis",
    "it_load_mw",
    "electrical_capacity_mw",
    "square_footage",
    "square_footage_basis",
    "rack_count",
    # Cooling
    "cooling_type",
    "cooling_technology",
    "pue",
    "pue_basis",
    "water_usage",
    # Electrical / power
    "power_source",
    "onsite_generation",
    "backup_power",
    "ups_type",
    "redundancy_tier",
    "power_redundancy",
    # Classification
    "dc_type",
    "segment",
    "is_ai_dc",
    # Ownership / misc
    "developer",
    "general_contractor",
    "tenants",
    "investment_usd",
    "jobs_created",
)

# Left null for now (deferred geocoding step).
DEFERRED_FIELDS: tuple[str, ...] = (
    "county",
    "county_fips",
    "latitude",
    "longitude",
)

# Computed provenance / run metadata (not LLM-extracted).
META_FIELDS: tuple[str, ...] = (
    "dc_id",
    "source",
    "source_record_id",
    "source_urls",
    "n_articles_used",
    "retrieved_at",
    "extracted_at",
    "model_used",
    "extraction_notes",
    "overall_confidence",
)

# Numeric fields (coerced with pd.to_numeric).
NUMERIC_FIELDS: frozenset[str] = frozenset(
    {
        "year_in_service",
        "capacity_mw",
        "it_load_mw",
        "electrical_capacity_mw",
        "square_footage",
        "rack_count",
        "pue",
        "investment_usd",
        "jobs_created",
        "latitude",
        "longitude",
    }
)

BOOL_FIELDS: frozenset[str] = frozenset({"is_ai_dc"})


def _provenance_columns(field: str) -> tuple[str, str]:
    return f"{field}_source_url", f"{field}_confidence"


def extraction_columns() -> tuple[str, ...]:
    """Full ordered column list for the wide output table."""
    cols: list[str] = ["dc_id"]
    cols.extend(IDENTITY_FIELDS)
    for f in EXTRACTED_FIELDS:
        src, conf = _provenance_columns(f)
        cols.extend([f, src, conf])
    cols.extend(DEFERRED_FIELDS)
    # META minus dc_id (already first)
    cols.extend(c for c in META_FIELDS if c != "dc_id")
    return tuple(cols)


EXTRACTION_FIELDS: tuple[str, ...] = extraction_columns()


def empty_extraction_df() -> pd.DataFrame:
    """Empty DataFrame with the full extraction column set."""
    return pd.DataFrame({c: pd.Series(dtype="object") for c in EXTRACTION_FIELDS})


# --- Coercion helpers (mirror lbl schema._clean_str / _coerce_enum) ----------

def _is_missing(v: object) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and pd.isna(v):
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _clean_str(v: object) -> object:
    if _is_missing(v):
        return pd.NA
    if isinstance(v, (list, tuple)):
        # Lists (e.g. tenants, source_urls) are kept as-is.
        return list(v)
    if not isinstance(v, str):
        v = str(v)
    v = v.strip()
    return v if v else pd.NA


def _coerce_enum(v: object, allowed: Iterable[str]) -> object:
    if _is_missing(v):
        return pd.NA
    s = str(v).strip()
    if s in allowed:  # segment values are case-sensitive multi-word strings
        return s
    s_low = s.lower()
    return s_low if s_low in allowed else "unknown"


def _coerce_bool(v: object) -> object:
    if _is_missing(v):
        return pd.NA
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "yes", "1", "y"):
        return True
    if s in ("false", "no", "0", "n"):
        return False
    return pd.NA


def conform(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder/fill columns to the full extraction schema and coerce types.

    Adds missing columns as NA, drops unknown columns, coerces numerics, booleans,
    and enum-like fields (unknown enum values become the ``unknown`` sentinel),
    and stamps the ``source`` column.
    """
    out = df.copy()
    missing = [c for c in EXTRACTION_FIELDS if c not in out.columns]
    if missing:
        # Add all missing columns at once to avoid DataFrame fragmentation.
        out = pd.concat(
            [out, pd.DataFrame({c: pd.NA for c in missing}, index=out.index)], axis=1
        )
    out = out[list(EXTRACTION_FIELDS)]

    for col in NUMERIC_FIELDS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in BOOL_FIELDS:
        if col in out.columns:
            out[col] = out[col].apply(_coerce_bool)

    for col, allowed in ENUM_FIELDS.items():
        if col in out.columns:
            out[col] = out[col].apply(_coerce_enum, allowed=allowed)
        conf_col = f"{col}_confidence"
        if conf_col in out.columns:
            out[conf_col] = out[conf_col].apply(_coerce_enum, allowed=CONFIDENCE_VALUES)

    # Confidence columns for non-enum extracted fields.
    for f in EXTRACTED_FIELDS:
        conf_col = f"{f}_confidence"
        if conf_col in out.columns:
            out[conf_col] = out[conf_col].apply(_coerce_enum, allowed=CONFIDENCE_VALUES)
    if "overall_confidence" in out.columns:
        out["overall_confidence"] = out["overall_confidence"].apply(
            _coerce_enum, allowed=CONFIDENCE_VALUES
        )

    out["source"] = out["source"].apply(lambda v: v if not _is_missing(v) else "dc_locations_search")

    # Clean remaining string-ish columns (skip numerics/bools/list columns).
    skip = set(NUMERIC_FIELDS) | set(BOOL_FIELDS) | {"source_urls", "tenants"}
    for col in EXTRACTION_FIELDS:
        if col in skip:
            continue
        out[col] = out[col].apply(_clean_str)

    return out.reset_index(drop=True)


def to_inventory_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Project the wide extraction table to the canonical ``INVENTORY_COLUMNS``.

    Returns a DataFrame that is a drop-in lbl-data-center-map source loader output
    (validated/normalized by the upstream ``conform``).
    """
    subset = pd.DataFrame(index=df.index)
    for col in INVENTORY_COLUMNS:
        if col in df.columns:
            subset[col] = df[col]
        else:
            subset[col] = pd.NA
    # source_record_id is the dc_id; source is stamped by upstream conform.
    if "source_record_id" in df.columns:
        subset["source_record_id"] = df["source_record_id"]
    if "extraction_notes" in df.columns:
        subset["notes"] = df["extraction_notes"]
    return inventory_conform(subset, source="dc_locations_search")


def flatten_extraction(parsed: dict[str, Any]) -> dict[str, Any]:
    """Flatten an LLM extraction dict into wide-table columns.

    The LLM returns, per extracted field, either ``null`` or an object
    ``{"value": ..., "source_url": ..., "confidence": ...}``. This expands each
    into ``<field>``, ``<field>_source_url``, ``<field>_confidence``.
    """
    row: dict[str, Any] = {}
    for field in EXTRACTED_FIELDS:
        src_col, conf_col = _provenance_columns(field)
        item = parsed.get(field)
        if isinstance(item, dict):
            row[field] = item.get("value")
            row[src_col] = item.get("source_url")
            row[conf_col] = item.get("confidence")
        elif item is None:
            row[field] = pd.NA
            row[src_col] = pd.NA
            row[conf_col] = pd.NA
        else:
            # Model returned a bare scalar; accept the value, no provenance.
            row[field] = item
            row[src_col] = pd.NA
            row[conf_col] = pd.NA
    return row
