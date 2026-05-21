# dc_locations_search

Given a CSV of data center **names + addresses**, this package gathers reference
articles about each location from the web (via the Tavily search API) and uses an
LLM (LBL's CBORG gateway) to extract structured attributes — cooling systems,
electrical/power infrastructure, construction status, capacity, classification —
into a provenance-tracked dataset.

Output conforms to the canonical `lbl-data-center-map` inventory schema and carries
the per-facility fields the `data-center-impact-model` consumes (`segment`, `pue`,
`cooling_type`, IT load MW, …). Every extracted value records the source URL it came
from and a confidence level; anything not stated in the gathered articles is left null.

## Install

```bash
conda create -n dc_locations_search python=3.12 -y
conda activate dc_locations_search
pip install -e /home/afhubbard/lbl-data-center-map   # canonical schema (not on PyPI)
pip install -e .
```

Copy `.env.example` to `.env` and set `CBORG_API_KEY` and `TAVILY_API_KEY`.

## Usage

```bash
# Validate the CSV + column mapping without any API calls
python -m dc_locations_search.cli validate-schema --input data/raw/sample.csv

# Dry run: print the queries that would run + config, no paid API calls
python -m dc_locations_search.cli extract --input data/raw/sample.csv --dry-run --limit 3

# Live extraction (first 3 rows)
python -m dc_locations_search.cli extract --input data/raw/sample.csv --limit 3 --max-workers 2

# Re-emit the canonical inventory subset from the latest run
python -m dc_locations_search.cli inventory-export
```

Flexible column mapping: if your CSV uses different headers, pass e.g.
`--name-col "DC Name" --city-col Town --state-col ST`.

## Outputs (`data/processed/`)

- `dc_attributes_<stamp>.{csv,parquet,xlsx}` — full wide table (one row per DC) with
  per-field `_source_url` / `_confidence` provenance columns.
- `inventory_subset_<stamp>.csv` — exactly the `lbl-data-center-map` `INVENTORY_COLUMNS`.
- `citations_<stamp>.csv` + `sources_<stamp>.jsonl` — per-value provenance sidecar.
- `dc_attributes_latest.parquet` — stable, updated each save.

Runs are resumable: a JSONL log keyed by a stable `dc_id` lets re-runs skip
already-extracted facilities. Raw search results are cached under
`data/interim/search_cache/`.

## Testing

```bash
pip install -r requirements-test.txt
pytest
```
