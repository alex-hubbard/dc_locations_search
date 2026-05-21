"""Typer CLI for dc_locations_search."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from loguru import logger

from dc_locations_search import config
from dc_locations_search.input_csv import ColumnMapping, load_input

app = typer.Typer(add_completion=False, help="Web-sourced data center attribute extraction.")


def _mapping(
    name_col: str, address_col: str, city_col: str, state_col: str,
    zip_col: str, country_col: str, operator_col: str,
) -> ColumnMapping:
    return ColumnMapping(
        name=name_col, address=address_col, city=city_col, state=state_col,
        zip=zip_col, country=country_col, operator=operator_col,
    )


@app.command()
def extract(
    input: Path = typer.Option(..., "--input", "-i", help="Input CSV of DC names + addresses."),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o"),
    name_col: str = typer.Option("name", "--name-col"),
    address_col: str = typer.Option("address", "--address-col"),
    city_col: str = typer.Option("city", "--city-col"),
    state_col: str = typer.Option("state", "--state-col"),
    zip_col: str = typer.Option("zip", "--zip-col"),
    country_col: str = typer.Option("country", "--country-col"),
    operator_col: str = typer.Option("operator", "--operator-col"),
    limit: Optional[int] = typer.Option(None, "--limit", "-n", help="Process only the first N rows."),
    max_workers: Optional[int] = typer.Option(None, "--max-workers"),
    save_every: Optional[int] = typer.Option(None, "--save-every"),
    dry_run: bool = typer.Option(False, "--dry-run", help="No paid API calls; print queries + config."),
    no_resume: bool = typer.Option(False, "--no-resume", help="Reprocess all rows, ignore the log."),
) -> None:
    """Gather articles and extract structured attributes for each data center."""
    from dc_locations_search.pipeline import run

    mapping = _mapping(name_col, address_col, city_col, state_col, zip_col, country_col, operator_col)
    result = run(
        input,
        mapping=mapping,
        output_dir=output_dir,
        limit=limit,
        max_workers=max_workers,
        resume=not no_resume,
        save_every=save_every,
        dry_run=dry_run,
    )
    typer.echo(
        f"Processed {result['processed']} | failed {result.get('failed', 0)}"
        + (" | DRY RUN" if result.get("dry_run") else "")
    )
    for label, path in (result.get("paths") or {}).items():
        typer.echo(f"  {label}: {path}")


@app.command("validate-schema")
def validate_schema(
    input: Path = typer.Option(..., "--input", "-i"),
    name_col: str = typer.Option("name", "--name-col"),
    address_col: str = typer.Option("address", "--address-col"),
    city_col: str = typer.Option("city", "--city-col"),
    state_col: str = typer.Option("state", "--state-col"),
    zip_col: str = typer.Option("zip", "--zip-col"),
    country_col: str = typer.Option("country", "--country-col"),
    operator_col: str = typer.Option("operator", "--operator-col"),
) -> None:
    """Validate the input CSV + column mapping without calling any API."""
    mapping = _mapping(name_col, address_col, city_col, state_col, zip_col, country_col, operator_col)
    df = load_input(input, mapping)
    typer.echo(f"OK: {len(df)} rows parsed.")
    typer.echo(f"Columns: {list(df.columns)}")
    typer.echo("Sample dc_ids:")
    for _, r in df.head(5).iterrows():
        typer.echo(f"  {r['dc_id']}  {r['facility_name']}")


@app.command("inventory-export")
def inventory_export(
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o"),
) -> None:
    """Re-emit the canonical INVENTORY_COLUMNS subset from the latest wide table."""
    import pandas as pd

    from dc_locations_search.schema import to_inventory_subset

    src = config.PROCESSED_DATA_DIR / "dc_attributes_latest.parquet"
    if not src.exists():
        raise typer.BadParameter(f"No latest table found at {src}. Run `extract` first.")
    df = pd.read_parquet(src)
    out_dir = Path(output_dir) if output_dir else config.PROCESSED_DATA_DIR
    out_path = out_dir / "inventory_subset_latest.csv"
    to_inventory_subset(df).to_csv(out_path, index=False)
    logger.info(f"Wrote {out_path}")
    typer.echo(str(out_path))


if __name__ == "__main__":
    app()
