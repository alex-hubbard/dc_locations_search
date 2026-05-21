from __future__ import annotations

import pandas as pd
import pytest

from dc_locations_search.input_csv import ColumnMapping, compute_dc_id, load_input


def test_default_mapping(sample_csv):
    df = load_input(sample_csv)
    assert len(df) == 3
    assert "facility_name" in df.columns
    assert "dc_id" in df.columns
    assert df["facility_name"].iloc[0] == "QTS Atlanta-Metro"


def test_overridden_mapping(tmp_path):
    p = tmp_path / "custom.csv"
    pd.DataFrame({"DC Name": ["Foo DC"], "Town": ["Reno"], "ST": ["NV"]}).to_csv(p, index=False)
    df = load_input(p, ColumnMapping(name="DC Name", city="Town", state="ST"))
    assert df["facility_name"].iloc[0] == "Foo DC"
    assert df["city"].iloc[0] == "Reno"


def test_missing_name_column_raises(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"address": ["123 St"]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="Required name column"):
        load_input(p)


def test_whitespace_stripped(tmp_path):
    p = tmp_path / "ws.csv"
    p.write_text("name,city\n  Spacey DC  ,  Reno  \n")
    df = load_input(p)
    assert df["facility_name"].iloc[0] == "Spacey DC"
    assert df["city"].iloc[0] == "Reno"


def test_empty_name_rows_dropped(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("name,city\nReal DC,Reno\n,Vegas\n")
    df = load_input(p)
    assert len(df) == 1


def test_dc_id_stable_and_unique(sample_csv):
    df1 = load_input(sample_csv)
    df2 = load_input(sample_csv)
    assert list(df1["dc_id"]) == list(df2["dc_id"])
    assert df1["dc_id"].nunique() == len(df1)


def test_dc_id_ignores_name_case_and_punctuation():
    a = compute_dc_id("QTS Atlanta-Metro", "250 Williams St", "Atlanta", "GA")
    b = compute_dc_id("qts atlanta metro", "250 williams st", "atlanta", "ga")
    assert a == b
