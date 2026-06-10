"""Tests for the file-handshake forced-tool path in the claude_code provider."""

from __future__ import annotations

import json
from pathlib import Path

from co_scientist.agents.schemas import RECORD_RESEARCH_PLAN_TOOL
from co_scientist.config import Config
from co_scientist.llm.anthropic_client import AgentCallSpec, CachedBlock
from co_scientist.llm.claude_code_client import ClaudeCodeClient
from co_scientist.llm.routing import ModelRoute


def _client() -> ClaudeCodeClient:
    return ClaudeCodeClient(Config(), db=None, budget=None)  # type: ignore[arg-type]


def _forced_spec() -> AgentCallSpec:
    return AgentCallSpec(
        route=ModelRoute(agent="parse_goal", mode="parse_goal", model="claude-sonnet-4-6"),
        system_blocks=[CachedBlock("sys", cache=True)],
        user_blocks=[CachedBlock("user", cache=False)],
        tools=[RECORD_RESEARCH_PLAN_TOOL],
        tool_choice={"type": "tool", "name": "record_research_plan"},
    )


def _auto_spec() -> AgentCallSpec:
    return AgentCallSpec(
        route=ModelRoute(agent="generation", mode="generation.literature", model="claude-opus-4-7"),
        system_blocks=[CachedBlock("sys", cache=True)],
        user_blocks=[CachedBlock("user", cache=False)],
        tools=[
            {"name": "web_search", "description": "", "input_schema": {"type": "object"}},
            RECORD_RESEARCH_PLAN_TOOL,
        ],
        tool_choice={"type": "auto"},
    )


def test_forced_tool_gets_tempfile_and_write_instructions() -> None:
    system_text, _user, schema, forced, tmp = _client()._build_prompts(_forced_spec())
    assert forced == "record_research_plan"
    assert schema == RECORD_RESEARCH_PLAN_TOOL["input_schema"]
    assert isinstance(tmp, Path)
    assert str(tmp) in system_text
    assert "Write tool" in system_text
    assert "COMMITTED record_research_plan" in system_text


def test_forced_tool_tempfile_unique_per_call() -> None:
    c = _client()
    _, _, _, _, tmp1 = c._build_prompts(_forced_spec())
    _, _, _, _, tmp2 = c._build_prompts(_forced_spec())
    assert tmp1 != tmp2


def test_auto_path_with_record_tool_gets_tempfile() -> None:
    system_text, _user, _schema, forced, tmp = _client()._build_prompts(_auto_spec())
    assert forced is None
    assert isinstance(tmp, Path)
    assert str(tmp) in system_text
    assert '"tool":' in system_text          # the {"tool", "payload"} wrapper
    assert "tool_calls" in system_text       # search tools keep the stdout JSON form


def test_adapt_prefers_file_data_for_forced_tool() -> None:
    c = _client()
    payload = {"objective": "X", "preferences": [], "idea_attributes": []}
    msg = c._adapt(
        {"result": "COMMITTED record_research_plan", "usage": {}},
        "claude-sonnet-4-6",
        forced_tool_name="record_research_plan",
        has_tools=True,
        file_data=payload,
        forced_tool_schema=RECORD_RESEARCH_PLAN_TOOL["input_schema"],
        tools=[RECORD_RESEARCH_PLAN_TOOL],
    )
    assert msg.stop_reason == "tool_use"
    tu = [b for b in msg.content if b.type == "tool_use"][0]
    assert tu.name == "record_research_plan"
    assert tu.input == payload


def test_adapt_unwraps_tool_payload_wrapper() -> None:
    c = _client()
    payload = {"objective": "X", "preferences": [], "idea_attributes": []}
    msg = c._adapt(
        {"result": "COMMITTED", "usage": {}},
        "claude-sonnet-4-6",
        forced_tool_name="record_research_plan",
        has_tools=True,
        file_data={"tool": "record_research_plan", "payload": payload},
        forced_tool_schema=RECORD_RESEARCH_PLAN_TOOL["input_schema"],
        tools=[RECORD_RESEARCH_PLAN_TOOL],
    )
    tu = [b for b in msg.content if b.type == "tool_use"][0]
    assert tu.input == payload


def test_adapt_auto_path_attributes_bare_payload_to_single_record_tool() -> None:
    c = _client()
    payload = {"objective": "X", "preferences": [], "idea_attributes": []}
    msg = c._adapt(
        {"result": "COMMITTED record_research_plan", "usage": {}},
        "claude-opus-4-7",
        forced_tool_name=None,
        has_tools=True,
        file_data=payload,
        forced_tool_schema=None,
        tools=_auto_spec().tools,
    )
    assert msg.stop_reason == "tool_use"
    tu = [b for b in msg.content if b.type == "tool_use"][0]
    assert tu.name == "record_research_plan"
    assert tu.input == payload


def test_adapt_auto_path_still_parses_stdout_tool_calls() -> None:
    c = _client()
    text = json.dumps({"tool_calls": [{"name": "web_search", "arguments": {"q": "x"}}]})
    msg = c._adapt(
        {"result": text, "usage": {}},
        "claude-opus-4-7",
        forced_tool_name=None,
        has_tools=True,
        file_data=None,
        forced_tool_schema=None,
        tools=_auto_spec().tools,
    )
    tu = [b for b in msg.content if b.type == "tool_use"][0]
    assert tu.name == "web_search"


def test_adapt_forced_falls_back_to_markdown_then_none() -> None:
    c = _client()
    md = (
        "prose...\n=== BEGIN ANSWER ===\n"
        "## objective\nX\n\n## preferences\n- a\n\n## idea_attributes\n- b\n"
        "=== END ANSWER ==="
    )
    msg = c._adapt(
        {"result": md, "usage": {}},
        "claude-sonnet-4-6",
        forced_tool_name="record_research_plan",
        has_tools=True,
        file_data=None,
        forced_tool_schema=RECORD_RESEARCH_PLAN_TOOL["input_schema"],
        tools=[RECORD_RESEARCH_PLAN_TOOL],
    )
    tu = [b for b in msg.content if b.type == "tool_use"][0]
    assert tu.input["objective"] == "X"
    # And with nothing recoverable at all → text block, no tool_use.
    msg2 = c._adapt(
        {"result": "just prose", "usage": {}},
        "claude-sonnet-4-6",
        forced_tool_name="record_research_plan",
        has_tools=True,
        file_data=None,
        forced_tool_schema=RECORD_RESEARCH_PLAN_TOOL["input_schema"],
        tools=[RECORD_RESEARCH_PLAN_TOOL],
    )
    assert all(b.type != "tool_use" for b in msg2.content)
