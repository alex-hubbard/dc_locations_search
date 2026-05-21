"""Project paths, logging, and tunable constants.

Path + loguru/tqdm wiring mirrors permit_data_extraction/permit_data_extraction/config.py
so the whole ecosystem behaves consistently.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env if present.
load_dotenv()

# --- Paths ---
PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

SEARCH_CACHE_DIR = INTERIM_DATA_DIR / "search_cache"
PROCESSING_LOG_PATH = INTERIM_DATA_DIR / "processing_log.jsonl"

# --- LLM (CBORG, OpenAI-compatible) ---
CBORG_BASE_URL = "https://api.cborg.lbl.gov"
# Primary extraction model: 1M context, ~$0.10/1M input tokens.
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash-lite")
# Fallback for documents that exceed the primary model's context window.
LLM_LARGE_MODEL = os.getenv("LLM_LARGE_MODEL", "gemini-2.0-flash-lite")
LLM_MAX_WORKERS = int(os.getenv("LLM_MAX_WORKERS", "4"))
LLM_MAX_OUTPUT_TOKENS = 32768
LLM_TEMPERATURE = 0.1
LLM_TIMEOUT_SECONDS = 60

# Retry / backoff (matches permit_data_extraction).
LLM_MAX_RETRIES = 5
LLM_BACKOFF_BASE = 2.0
LLM_BACKOFF_MAX = 60.0

# Chunking: aggregated article text above this many chars is split.
DEFAULT_MAX_CHUNK_CHARS = 120_000

# --- Search (Tavily by default; Serper alternate) ---
SEARCH_BACKEND = os.getenv("SEARCH_BACKEND", "tavily")  # tavily | serper
TAVILY_MAX_RESULTS = int(os.getenv("TAVILY_MAX_RESULTS", "6"))
TAVILY_SEARCH_DEPTH = "advanced"
# Minimum usable results from the primary query before we issue secondary queries.
SEARCH_MIN_PRIMARY_RESULTS = 3
# Per-article character budget when building the LLM context block.
PER_ARTICLE_CHAR_BUDGET = 8_000
# Total aggregated-context character budget (soft; chunking handles overflow).
TOTAL_CONTEXT_CHAR_BUDGET = 100_000

# Light domain deny-list: low-signal aggregators / forums. Kept short so we
# don't over-filter legitimate operator and trade-press coverage.
SEARCH_DENY_DOMAINS = frozenset(
    {
        "pinterest.com",
        "facebook.com",
        "twitter.com",
        "x.com",
        "reddit.com",
        "quora.com",
        "youtube.com",
    }
)

# --- Persistence ---
SAVE_EVERY_N = int(os.getenv("SAVE_EVERY_N", "25"))

# The canonical `source` identifier stamped on every output row.
SOURCE_ID = "dc_locations_search"


def _env_flag(name: str, default: bool = True) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


# Importing lbl_data_center_map triggers its own unguarded `logger.remove(0)`.
# Do it here, while loguru's default handler still exists, so that import never
# races our own handler teardown below and raises "no existing handler".
try:
    import lbl_data_center_map  # noqa: F401
except ImportError:
    pass

# If tqdm is installed, route loguru through tqdm.write so progress bars and
# logs don't clobber each other (https://github.com/Delgan/loguru/issues/135).
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except (ModuleNotFoundError, ValueError):
    pass


def ensure_dirs() -> None:
    """Create the data directories if they don't exist."""
    for d in (RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, SEARCH_CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)
