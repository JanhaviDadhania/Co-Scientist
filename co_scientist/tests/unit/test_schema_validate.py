"""Unit tests for the forced-tool payload validator."""

from __future__ import annotations

from co_scientist.agents.schemas import (
    RECORD_HYPOTHESIS_TOOL,
    RECORD_RESEARCH_PLAN_TOOL,
)
from co_scientist.llm.schema_validate import validate_payload

PLAN_SCHEMA = RECORD_RESEARCH_PLAN_TOOL["input_schema"]
HYP_SCHEMA = RECORD_HYPOTHESIS_TOOL["input_schema"]


def test_valid_plan_payload_passes() -> None:
    payload = {
        "objective": "Investigate X",
        "preferences": ["testable", "specific"],
        "idea_attributes": ["mechanistic"],
        "n_ideas": 15,
    }
    assert validate_payload(PLAN_SCHEMA, payload) == []


def test_non_dict_payload_fails() -> None:
    errs = validate_payload(PLAN_SCHEMA, ["not", "an", "object"])
    assert errs and "JSON object" in errs[0]


def test_missing_required_field_reported_by_name() -> None:
    errs = validate_payload(PLAN_SCHEMA, {"objective": "X", "preferences": []})
    assert any("idea_attributes" in e for e in errs)


def test_empty_required_field_reported() -> None:
    errs = validate_payload(
        PLAN_SCHEMA,
        {"objective": "", "preferences": [], "idea_attributes": []},
    )
    assert any("objective" in e and "empty" in e for e in errs)


def test_wrong_type_reported() -> None:
    errs = validate_payload(
        PLAN_SCHEMA,
        {"objective": "X", "preferences": "not-a-list", "idea_attributes": []},
    )
    assert any("preferences" in e and "array" in e for e in errs)


def test_n_ideas_must_be_integer() -> None:
    errs = validate_payload(
        PLAN_SCHEMA,
        {"objective": "X", "preferences": [], "idea_attributes": [], "n_ideas": "ten"},
    )
    assert any("n_ideas" in e for e in errs)


def test_extra_fields_tolerated() -> None:
    payload = {
        "objective": "X", "preferences": [], "idea_attributes": [],
        "something_extra": 42,
    }
    assert validate_payload(PLAN_SCHEMA, payload) == []


def test_array_of_objects_required_keys() -> None:
    payload = {
        "title": "T", "statement": "S", "mechanism": "M",
        "entities": ["a"], "anticipated_outcomes": "O",
        "novelty_argument": "N",
        "citations": [{"url": "https://x", "title": "ok"}, {"title": "no url"}],
    }
    errs = validate_payload(HYP_SCHEMA, payload)
    assert any("citations[1]" in e and "url" in e for e in errs)


def test_enum_membership_checked() -> None:
    payload = {
        "title": "T", "statement": "S", "mechanism": "M",
        "entities": ["a"], "anticipated_outcomes": "O",
        "novelty_argument": "N", "citations": [],
        "strategy": "not_a_strategy",
    }
    errs = validate_payload(HYP_SCHEMA, payload)
    assert any("strategy" in e for e in errs)
