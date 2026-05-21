"""CBORG (OpenAI-compatible) LLM client and the extraction cascade.

Ported from permit_data_extraction/permit_data_extraction/dataset.py: the client
setup, the retry loop with exponential backoff, token-limit detection, JSON
fence-stripping, and the full-doc -> chunked -> large-model fallback cascade.
Adapted to our extraction shape (a flat dict of per-field objects).
"""

from __future__ import annotations

import json
import logging
import random
import time
from typing import Any, Optional

import openai
from dotenv import dotenv_values
from loguru import logger

from dc_locations_search import config
from dc_locations_search.prompt import build_prompt

_SYSTEM_PROMPT = (
    "You are an expert at extracting structured information about data centers "
    "from reference articles. Always respond with valid JSON. Only use facts "
    "stated in the provided text; use null for anything not stated."
)

# Confidence rank used when merging chunk results.
_CONF_RANK = {"high": 3, "medium": 2, "low": 1}


def configure_llm() -> openai.OpenAI:
    """Build an OpenAI client pointed at the CBORG gateway.

    Reads CBORG_API_KEY from .env (matching permit_data_extraction line 41).
    Performs a lightweight connectivity check.
    """
    api_key = dotenv_values().get("CBORG_API_KEY") or ""
    if not api_key:
        raise RuntimeError("CBORG_API_KEY is not set in the .env file")
    client = openai.OpenAI(api_key=api_key, base_url=config.CBORG_BASE_URL)
    try:
        client.models.list()
    except Exception as e:  # noqa: BLE001 - connectivity check is best-effort
        logger.warning(f"CBORG connectivity check failed (continuing anyway): {e}")
    return client


def _is_token_limit_error(error: object) -> bool:
    """Heuristics for token/context-limit errors (verbatim from permit project)."""
    message = str(error).lower()
    return (
        "maximum context length" in message
        or "context length" in message
        or "context window" in message
        or "token limit" in message
        or "too many tokens" in message
        or ("context" in message and "token" in message)
        or "max_tokens must be at least 1" in message
        or ("max_tokens" in message and "got -" in message)
        or "finish_reason=length" in message
    )


def _is_retryable_error(error: object) -> bool:
    """Rate-limit / transient server errors are retryable (from permit project)."""
    error_str = str(error).lower()
    status = getattr(error, "status_code", None)
    if status in (429, 500, 502, 503, 504):
        return True
    return (
        "rate limit" in error_str
        or "429" in error_str
        or "502" in error_str
        or "503" in error_str
        or "server error" in error_str
        or "overloaded" in error_str
    )


def _parse_json_response(content: str) -> dict[str, Any]:
    """Strip ```json fences and parse. Raises json.JSONDecodeError on failure."""
    text = content.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    data = json.loads(text)
    if not isinstance(data, dict):
        raise json.JSONDecodeError("Top-level JSON is not an object", text, 0)
    return data


def _invoke_llm_for_model(
    client: openai.OpenAI, prompt: str, label: str, model_name: str
) -> tuple[Optional[dict], Optional[str]]:
    """Call the LLM with one model and parse JSON. Returns (data, error_str).

    Retries with exponential backoff on rate-limit/transient errors.
    """
    time.sleep(0.1)  # small stagger to avoid thundering herd
    last_error = None
    response = None
    for attempt in range(1, config.LLM_MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=config.LLM_TEMPERATURE,
                max_tokens=config.LLM_MAX_OUTPUT_TOKENS,
                timeout=config.LLM_TIMEOUT_SECONDS,
                response_format={"type": "json_object"},
            )
            break
        except Exception as e:  # noqa: BLE001
            last_error = e
            if _is_retryable_error(e) and attempt < config.LLM_MAX_RETRIES:
                delay = min(
                    config.LLM_BACKOFF_BASE**attempt + random.uniform(0, 1),
                    config.LLM_BACKOFF_MAX,
                )
                logger.warning(
                    f"Retryable error for {label} (attempt {attempt}/{config.LLM_MAX_RETRIES}): "
                    f"{e}. Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
                continue
            logging.error(f"LLM API call failed for {label} ({model_name}): {e}")
            return None, str(e)

    if response is None:
        return None, str(last_error) if last_error else "No response"
    if not response.choices:
        return None, "Empty or malformed response"

    choice = response.choices[0]
    content = getattr(choice.message, "content", None)
    finish_reason = getattr(choice, "finish_reason", "unknown")
    if content is None:
        return None, f"Empty content (finish_reason={finish_reason})"
    if finish_reason == "length":
        # Output truncated; surface as token-limit so chunking picks it up.
        return None, "Output truncated (finish_reason=length)"

    try:
        return _parse_json_response(content), None
    except json.JSONDecodeError as e:
        return None, f"JSON decode error: {e}"


def _split_text_into_chunks(text: str, max_chunk_chars: int) -> list[str]:
    """Split text into <= max_chunk_chars chunks, preferring SOURCE boundaries."""
    if len(text) <= max_chunk_chars:
        return [text]
    chunks: list[str] = []
    blocks = text.split("[[SOURCE ")
    current = ""
    for i, block in enumerate(blocks):
        piece = block if i == 0 else "[[SOURCE " + block
        if current and len(current) + len(piece) > max_chunk_chars:
            chunks.append(current)
            current = piece
        else:
            current += piece
        # A single block bigger than the budget: hard-split it.
        while len(current) > max_chunk_chars:
            chunks.append(current[:max_chunk_chars])
            current = current[max_chunk_chars:]
    if current:
        chunks.append(current)
    return chunks


def _merge_extractions(results: list[dict]) -> dict[str, Any]:
    """Merge per-chunk extraction dicts.

    For each field, keep the populated value with the highest confidence (ties
    broken by first-seen). Field values are the {"value","source_url","confidence"}
    objects produced by the prompt.
    """
    merged: dict[str, Any] = {}
    for res in results:
        for key, item in res.items():
            if item is None:
                continue
            if key in ("overall_confidence", "extraction_notes"):
                merged.setdefault(key, item)
                continue
            existing = merged.get(key)
            if existing is None:
                merged[key] = item
                continue
            new_conf = _CONF_RANK.get((item or {}).get("confidence", "low"), 0) if isinstance(item, dict) else 0
            old_conf = _CONF_RANK.get((existing or {}).get("confidence", "low"), 0) if isinstance(existing, dict) else 0
            if new_conf > old_conf:
                merged[key] = item
    return merged


def _extract_chunked(
    client: openai.OpenAI, dc_header: str, article_text: str, label: str,
    model_name: str, max_chunk_chars: int,
) -> tuple[Optional[dict], Optional[str]]:
    chunks = _split_text_into_chunks(article_text, max_chunk_chars)
    logger.info(f"Chunked extraction for {label}: {len(chunks)} chunk(s)")
    results: list[dict] = []
    last_error: Optional[str] = None
    for i, chunk in enumerate(chunks, 1):
        prompt = build_prompt(dc_header, chunk)
        data, err = _invoke_llm_for_model(client, prompt, f"{label}_chunk{i}", model_name)
        if data is not None:
            results.append(data)
        else:
            last_error = err
    if not results:
        return None, last_error
    return _merge_extractions(results), None


def extract(
    client: openai.OpenAI,
    dc_header: str,
    article_text: str,
    label: str,
    *,
    max_chunk_chars: int | None = None,
    allow_large_model_retry: bool = True,
) -> tuple[Optional[dict], str]:
    """Run the extraction cascade for one data center.

    1. Full context with the primary model.
    2. On token-limit, split into chunks and merge (same model).
    3. Large-model fallback on persistent token-limit errors.

    Returns (parsed_dict | None, model_used_label).
    """
    if not client or not article_text:
        return None, config.LLM_MODEL

    chunk_size = max_chunk_chars or config.DEFAULT_MAX_CHUNK_CHARS
    prompt = build_prompt(dc_header, article_text)

    data, error_str = _invoke_llm_for_model(client, prompt, label, config.LLM_MODEL)
    if data is not None:
        return data, config.LLM_MODEL

    if error_str and _is_token_limit_error(error_str):
        logger.warning(f"Token limit for {label}; splitting into chunks.")
        data, chunk_err = _extract_chunked(
            client, dc_header, article_text, label, config.LLM_MODEL, chunk_size
        )
        if data is not None:
            return data, f"{config.LLM_MODEL} (chunked)"
        error_str = chunk_err or error_str

    if (
        allow_large_model_retry
        and config.LLM_LARGE_MODEL != config.LLM_MODEL
        and error_str
        and _is_token_limit_error(error_str)
    ):
        logger.warning(f"Retrying {label} with large model {config.LLM_LARGE_MODEL}.")
        data, _ = _invoke_llm_for_model(client, prompt, label, config.LLM_LARGE_MODEL)
        if data is not None:
            return data, config.LLM_LARGE_MODEL

    if "api key not valid" in (error_str or "").lower():
        logger.error("Hint: check your CBORG_API_KEY.")
    return None, config.LLM_MODEL
