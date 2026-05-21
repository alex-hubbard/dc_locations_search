"""Load and validate the input CSV of data center names + addresses.

Supports flexible column mapping: the user's CSV can use any column names, which
are mapped to the canonical internal names (facility_name, address, city, ...).
Each row gets a stable ``dc_id`` derived from the normalized name + address, so
re-running with the same CSV resumes cleanly regardless of row order.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from dc_locations_search.schema import normalize_address, normalize_facility_name

# Internal canonical names the rest of the pipeline expects.
CANONICAL_NAME = "facility_name"
CANONICAL_ADDRESS = "address"
CANONICAL_CITY = "city"
CANONICAL_STATE = "state"
CANONICAL_ZIP = "zip_code"
CANONICAL_COUNTRY = "country"
CANONICAL_OPERATOR = "operator"


@dataclass
class ColumnMapping:
    """Maps user CSV column names -> canonical internal fields.

    Only ``name`` is required to exist in the input; the rest are optional and
    silently skipped when absent.
    """

    name: str = "name"
    address: str = "address"
    city: str = "city"
    state: str = "state"
    zip: str = "zip"
    country: str = "country"
    operator: str = "operator"

    def as_pairs(self) -> dict[str, str]:
        """user_column -> canonical_internal_field."""
        return {
            self.name: CANONICAL_NAME,
            self.address: CANONICAL_ADDRESS,
            self.city: CANONICAL_CITY,
            self.state: CANONICAL_STATE,
            self.zip: CANONICAL_ZIP,
            self.country: CANONICAL_COUNTRY,
            self.operator: CANONICAL_OPERATOR,
        }


def compute_dc_id(name: object, address: object, city: object, state: object) -> str:
    """Stable 16-char id from normalized facility name + address.

    Uses the lbl-data-center-map normalization helpers so the key matches how
    that project dedupes facilities.
    """
    key = f"{normalize_facility_name(name)}|{normalize_address(address, city, state)}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def load_input(path: str | Path, mapping: ColumnMapping | None = None) -> pd.DataFrame:
    """Read the input CSV, apply column mapping, validate, and add ``dc_id``.

    Returns a DataFrame with canonical columns
    (facility_name, address, city, state, zip_code, country, operator, dc_id).
    Raises ``ValueError`` if the mapped name column is missing or no rows remain.
    """
    mapping = mapping or ColumnMapping()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip() for c in df.columns]

    if mapping.name not in df.columns:
        raise ValueError(
            f"Required name column '{mapping.name}' not found in {path.name}. "
            f"Available columns: {list(df.columns)}. "
            f"Use the --name-col option to point at the right column."
        )

    pairs = {user: canon for user, canon in mapping.as_pairs().items() if user in df.columns}
    out = df.rename(columns=pairs)[list(pairs.values())].copy()

    # Strip whitespace on all string cells.
    for col in out.columns:
        out[col] = out[col].apply(lambda v: v.strip() if isinstance(v, str) else v)

    # Ensure all canonical columns exist (missing optional ones -> NA).
    for canon in (
        CANONICAL_NAME,
        CANONICAL_ADDRESS,
        CANONICAL_CITY,
        CANONICAL_STATE,
        CANONICAL_ZIP,
        CANONICAL_COUNTRY,
        CANONICAL_OPERATOR,
    ):
        if canon not in out.columns:
            out[canon] = pd.NA

    # Drop rows with no facility name.
    out = out[out[CANONICAL_NAME].notna() & (out[CANONICAL_NAME].astype(str).str.strip() != "")]
    out = out.reset_index(drop=True)
    if out.empty:
        raise ValueError(f"No usable rows (with a non-empty name) in {path.name}.")

    out["dc_id"] = [
        compute_dc_id(
            r[CANONICAL_NAME], r[CANONICAL_ADDRESS], r[CANONICAL_CITY], r[CANONICAL_STATE]
        )
        for _, r in out.iterrows()
    ]
    return out
