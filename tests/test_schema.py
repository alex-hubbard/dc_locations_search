from __future__ import annotations

import pandas as pd

from dc_locations_search import schema
from dc_locations_search.schema import (
    EXTRACTION_FIELDS,
    INVENTORY_COLUMNS,
    SEGMENT_VALUES,
    conform,
    flatten_extraction,
    to_inventory_subset,
)


def test_extraction_columns_include_provenance():
    assert "cooling_type" in EXTRACTION_FIELDS
    assert "cooling_type_source_url" in EXTRACTION_FIELDS
    assert "cooling_type_confidence" in EXTRACTION_FIELDS
    assert EXTRACTION_FIELDS[0] == "dc_id"


def test_conform_fills_missing_and_stamps_source():
    df = pd.DataFrame([{"dc_id": "abc", "facility_name": "Foo"}])
    out = conform(df)
    assert set(EXTRACTION_FIELDS).issubset(out.columns)
    assert out["source"].iloc[0] == "dc_locations_search"


def test_conform_coerces_bad_enum_to_unknown():
    df = pd.DataFrame([{"dc_id": "a", "cooling_type": "magic_cooling", "status": "live"}])
    out = conform(df)
    assert out["cooling_type"].iloc[0] == "unknown"
    assert out["status"].iloc[0] == "unknown"


def test_conform_keeps_valid_enum_and_segment():
    df = pd.DataFrame([{"dc_id": "a", "cooling_type": "liquid_cooled", "segment": "Internet Giants"}])
    out = conform(df)
    assert out["cooling_type"].iloc[0] == "liquid_cooled"
    assert out["segment"].iloc[0] == "Internet Giants"


def test_conform_coerces_numeric():
    df = pd.DataFrame([{"dc_id": "a", "capacity_mw": "72", "pue": "1.3"}])
    out = conform(df)
    assert out["capacity_mw"].iloc[0] == 72.0
    assert out["pue"].iloc[0] == 1.3


def test_to_inventory_subset_has_exact_columns():
    df = conform(pd.DataFrame([{"dc_id": "a", "facility_name": "Foo", "capacity_mw": 10}]))
    inv = to_inventory_subset(df)
    assert list(inv.columns) == list(INVENTORY_COLUMNS)
    assert inv["source"].iloc[0] == "dc_locations_search"


def test_segment_values_match_impact_model_order_lc():
    # Drift guard: first entry is the AI segment, count is 10.
    assert SEGMENT_VALUES[0] == "Liquid Cooled AI"
    assert len(SEGMENT_VALUES) == 10


def test_flatten_extraction_expands_field_objects():
    parsed = {
        "cooling_type": {"value": "air_cooled", "source_url": "http://x", "confidence": "high"},
        "pue": None,
    }
    row = flatten_extraction(parsed)
    assert row["cooling_type"] == "air_cooled"
    assert row["cooling_type_source_url"] == "http://x"
    assert row["cooling_type_confidence"] == "high"
    assert pd.isna(row["pue"])


def test_enums_imported_from_lbl():
    # Ensures the editable lbl dependency is the source of truth.
    assert "operational" in schema.STATUS_VALUES
    assert "hyperscale" in schema.DC_TYPE_VALUES
    assert "it_load" in schema.CAPACITY_BASIS_VALUES
