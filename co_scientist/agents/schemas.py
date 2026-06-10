"""Shared Anthropic tool-use schemas for structured outputs.

Each agent gets one or more of these as required tool calls. Using tool-use
schemas (rather than "respond in JSON") is the most reliable structured-output
mechanism on the Anthropic API.
"""

from __future__ import annotations

from typing import Any

RECORD_HYPOTHESIS_TOOL: dict[str, Any] = {
    "name": "record_hypothesis",
    "description": (
        "Record a structured hypothesis at the end of generation/evolution. Call this "
        "exactly once when your hypothesis is finalized. All citations must reference "
        "URLs that previously appeared in your tool_result outputs from search/fetch."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "title":     {"type": "string", "description": "Short noun-phrase title."},
            "statement": {"type": "string", "description": "One sentence: the hypothesis."},
            "mechanism": {"type": "string", "description": "Detailed causal/mechanistic story."},
            "entities": {
                "type": "array", "items": {"type": "string"},
                "description": "Specific named actors (proteins, materials, datasets, agents, etc.).",
            },
            "anticipated_outcomes": {
                "type": "string",
                "description": "What would be observed if the hypothesis is true.",
            },
            "novelty_argument": {
                "type": "string",
                "description": "What is new relative to the cited literature.",
            },
            "citations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "url":     {"type": "string"},
                        "title":   {"type": "string"},
                        "excerpt": {"type": "string", "description": "Verbatim short quote from the source."},
                        "doi":     {"type": "string"},
                        "year":    {"type": "integer"},
                    },
                    "required": ["url", "title"],
                },
            },
            "strategy": {
                "type": "string",
                "enum": ["literature", "debate", "combine", "simplify",
                         "out_of_box", "feasibility", "assumption", "feedback_driven"],
                "description": "Strategy that produced this hypothesis (set by the agent).",
            },
            "parent_ids": {
                "type": "array", "items": {"type": "string"},
                "description": "Hypothesis IDs this one descends from (Evolution only).",
            },
            "creative_work": {
                "type": "string",
                "description": (
                    "The creative warm-up (poem / very short story) written "
                    "before hypothesizing. Stored as a creative artifact and "
                    "fed to the out-of-the-box evolution step."
                ),
            },
        },
        "required": [
            "title", "statement", "mechanism",
            "entities", "anticipated_outcomes", "novelty_argument", "citations",
        ],
    },
}


RECORD_REVIEW_TOOL: dict[str, Any] = {
    "name": "record_review",
    "description": (
        "Record a structured review of a hypothesis. Every claim in `evidence[]` "
        "must include a URL and a verbatim excerpt; the URL must have appeared in "
        "your tool_result outputs. Pick exactly one verdict."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": [
                    "already_explained",
                    "other_more_likely",
                    "missing_piece",
                    "neutral",
                    "disproved",
                ],
            },
            "kind": {
                "type": "string",
                "enum": ["full", "verification", "observation", "simulation"],
                "description": "Which review mode you ran.",
            },
            "novelty":     {"type": "number", "minimum": 0, "maximum": 1},
            "correctness": {"type": "number", "minimum": 0, "maximum": 1},
            "testability": {"type": "number", "minimum": 0, "maximum": 1},
            "feasibility": {"type": "number", "minimum": 0, "maximum": 1},
            "assumptions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "assumption":   {"type": "string"},
                        "plausibility": {"type": "string", "enum": ["plausible", "uncertain", "implausible"]},
                        "rationale":    {"type": "string"},
                    },
                    "required": ["assumption", "plausibility", "rationale"],
                },
            },
            "evidence": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim":   {"type": "string"},
                        "url":     {"type": "string"},
                        "excerpt": {"type": "string"},
                    },
                    "required": ["claim", "url", "excerpt"],
                },
            },
            "notes": {"type": "string", "description": "Anything that didn't fit the structured fields."},
        },
        "required": ["verdict", "kind", "evidence"],
    },
}


RECORD_SYSTEM_FEEDBACK_TOOL: dict[str, Any] = {
    "name": "record_system_feedback",
    "description": "Record a structured meta-review of the session's reviews + debates.",
    "input_schema": {
        "type": "object",
        "properties": {
            "common_weaknesses":     {"type": "array", "items": {"type": "string"}},
            "common_strengths":      {"type": "array", "items": {"type": "string"}},
            "suggested_focus_areas": {"type": "array", "items": {"type": "string"}},
            "narrative":             {"type": "string"},
        },
        "required": ["narrative"],
    },
}


RECORD_RESEARCH_PLAN_TOOL: dict[str, Any] = {
    "name": "record_research_plan",
    "description": "Record the parsed research plan derived from the scientist's goal.",
    "input_schema": {
        "type": "object",
        "properties": {
            "objective":       {"type": "string"},
            "preferences":     {"type": "array", "items": {"type": "string"}},
            "constraints":     {"type": "array", "items": {"type": "string"}},
            "idea_attributes": {"type": "array", "items": {"type": "string"}},
            "domain_hint":     {"type": "string"},
            "notes":           {"type": "string"},
            "n_ideas": {
                "type": "integer",
                "description": (
                    "How many hypotheses this run should generate. If the "
                    "discussion or goal states a desired number of ideas, use "
                    "it; otherwise use 15."
                ),
            },
        },
        "required": ["objective", "preferences", "idea_attributes"],
    },
}
