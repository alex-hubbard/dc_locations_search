"""dc_locations_search: web-sourced data center attribute extraction.

Given a CSV of data center names + addresses, gather reference articles from the
web (Tavily) and extract structured attributes (cooling, electrical/power
infrastructure, construction status, capacity, ...) with an LLM (CBORG), into a
provenance-tracked dataset that conforms to the lbl-data-center-map inventory
schema and feeds the data-center-impact-model.
"""

__version__ = "0.0.1"
