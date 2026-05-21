"""LLM extraction prompt.

The prompt instructs the model to extract a fixed set of fields ONLY from the
supplied article text, returning per-field objects with value + source_url +
confidence, and ``null`` for anything not stated. The aggregated context blocks
are labeled ``[[SOURCE n: <url> | <title>]]`` so the model can cite the URL each
value came from.
"""

from __future__ import annotations

from dc_locations_search.schema import (
    COOLING_TYPE_VALUES,
    DC_TYPE_VALUES,
    PUE_BASIS_VALUES,
    REDUNDANCY_TIER_VALUES,
    SEGMENT_VALUES,
    SQFT_BASIS_VALUES,
    STATUS_VALUES,
)

# Field-by-field instructions. Enums are spelled out so the model uses the exact
# vocabulary the downstream schema expects.
_FIELD_SPEC = f"""
Identity / ownership:
- operator: company that operates/runs the facility (string)
- company: owner / parent company (string)

Status & construction:
- status: one of {sorted(STATUS_VALUES)}
- construction_phase: free text, e.g. "Phase 2 of 4" (string)
- year_in_service: 4-digit year the facility became (or is expected) operational (integer)
- announced_date: ISO date (YYYY-MM-DD) the project was announced (string)
- expected_completion: ISO date (YYYY-MM-DD) of expected completion (string)

Capacity & size:
- capacity_mw: single headline capacity in megawatts (number)
- capacity_basis: what capacity_mw measures, one of ["ups","it_load","demand","queue","nameplate","unknown"]
- it_load_mw: IT load in MW, only if separately stated (number)
- electrical_capacity_mw: total electrical/utility capacity in MW (number)
- square_footage: facility size in square feet (number)
- square_footage_basis: one of {sorted(SQFT_BASIS_VALUES)}
- rack_count: number of racks (integer)

Cooling:
- cooling_type: one of {sorted(COOLING_TYPE_VALUES)}
- cooling_technology: free text, e.g. "direct-to-chip liquid", "evaporative", "chilled water CRAH" (string)
- pue: Power Usage Effectiveness as a number (number)
- pue_basis: one of {sorted(PUE_BASIS_VALUES)}
- water_usage: water use / WUE description or value (string)

Electrical / power:
- power_source: utility or grid serving the site (string)
- onsite_generation: on-site generation, e.g. "gas turbines", "fuel cells", "solar" (string)
- backup_power: backup, e.g. "diesel gensets", "BESS" (string)
- ups_type: UPS technology (string)
- redundancy_tier: one of {sorted(REDUNDANCY_TIER_VALUES)}
- power_redundancy: redundancy configuration, e.g. "N+1", "2N" (string)

Classification:
- dc_type: one of {sorted(DC_TYPE_VALUES)}
- segment: one of {list(SEGMENT_VALUES)} (ONLY if the article is explicit about operator type; else null)
- is_ai_dc: true if this is an AI/GPU training or inference facility, false if explicitly not, else null (boolean)

Ownership / misc:
- developer: developer of the project (string)
- general_contractor: general contractor / builder (string)
- tenants: list of named tenants/customers (array of strings)
- investment_usd: announced investment in US dollars (number)
- jobs_created: number of jobs created (integer)
""".strip()

PROMPT_TEMPLATE = f"""You are an expert analyst extracting structured facts about a specific data center \
from reference articles. You will be given the data center's name/address and the text of \
several web articles, each prefixed by a marker like [[SOURCE 1: <url> | <title>]].

CRITICAL RULES:
1. Extract ONLY facts that are explicitly stated in the provided article text. Do NOT use outside \
knowledge, do NOT guess, and do NOT infer values that are not written down.
2. If a field is not stated in the articles, set it to null. Leaving fields null is expected and correct.
3. For every field you DO fill, you must cite the URL of the source it came from (use the URL from the \
nearest [[SOURCE n: <url> ...]] marker) and a confidence of "high", "medium", or "low".
4. Make sure values match the article (correct facility, not a different site by the same operator).

Return a single JSON object. For EACH field below, the value must be either null OR an object of the \
form: {{"value": <the value>, "source_url": "<url it came from>", "confidence": "high|medium|low"}}.
Also include a top-level "overall_confidence" ("high"|"medium"|"low") and "extraction_notes" (a short \
string of caveats, or null).

Fields to extract:
{_FIELD_SPEC}

Data center under analysis:
{{dc_header}}

Article text:
{{article_text}}
"""


def build_prompt(dc_header: str, article_text: str) -> str:
    """Fill the template. Uses str.replace because the template contains JSON braces."""
    return PROMPT_TEMPLATE.replace("{dc_header}", dc_header).replace(
        "{article_text}", article_text
    )
