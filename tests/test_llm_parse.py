from __future__ import annotations

import json

import pytest

from dc_locations_search.llm import (
    _is_retryable_error,
    _is_token_limit_error,
    _merge_extractions,
    _parse_json_response,
    _split_text_into_chunks,
)


def test_parse_clean_json():
    assert _parse_json_response('{"a": 1}') == {"a": 1}


def test_parse_json_fence():
    assert _parse_json_response('```json\n{"a": 1}\n```') == {"a": 1}


def test_parse_bare_fence():
    assert _parse_json_response('```\n{"a": 1}\n```') == {"a": 1}


def test_parse_non_object_raises():
    with pytest.raises(json.JSONDecodeError):
        _parse_json_response("[1, 2, 3]")


@pytest.mark.parametrize(
    "msg",
    ["maximum context length exceeded", "token limit reached", "finish_reason=length",
     "max_tokens must be at least 1"],
)
def test_token_limit_detection_true(msg):
    assert _is_token_limit_error(msg)


def test_token_limit_detection_false():
    assert not _is_token_limit_error("some unrelated error")


@pytest.mark.parametrize("msg", ["rate limit exceeded", "429 too many", "503 unavailable", "overloaded"])
def test_retryable_true(msg):
    assert _is_retryable_error(msg)


def test_retryable_false():
    assert not _is_retryable_error("400 bad request")


def test_retryable_status_code_attr():
    class Err(Exception):
        status_code = 429

    assert _is_retryable_error(Err())


def test_split_short_text_single_chunk():
    assert _split_text_into_chunks("hello", 1000) == ["hello"]


def test_split_on_source_boundaries():
    text = "[[SOURCE 1: a]]\n" + "x" * 50 + "[[SOURCE 2: b]]\n" + "y" * 50
    chunks = _split_text_into_chunks(text, 60)
    assert len(chunks) >= 2
    assert "".join(chunks) == text


def test_merge_prefers_higher_confidence():
    a = {"pue": {"value": 1.5, "confidence": "low"}}
    b = {"pue": {"value": 1.3, "confidence": "high"}}
    merged = _merge_extractions([a, b])
    assert merged["pue"]["value"] == 1.3


def test_merge_fills_nulls():
    a = {"pue": None, "cooling_type": {"value": "air_cooled", "confidence": "high"}}
    b = {"pue": {"value": 1.3, "confidence": "medium"}}
    merged = _merge_extractions([a, b])
    assert merged["pue"]["value"] == 1.3
    assert merged["cooling_type"]["value"] == "air_cooled"
