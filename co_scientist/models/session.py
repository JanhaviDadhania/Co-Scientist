"""Session and parsed-research-plan models."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

SessionStatus = Literal["running", "paused", "done", "failed", "aborted"]


class ResearchPlan(BaseModel):
    """The Supervisor's parsed view of the scientist's goal.

    Filled by a one-shot Claude call at session start (`parse_goal`).
    """

    objective: str
    preferences: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    idea_attributes: list[str] = Field(default_factory=list)
    domain_hint: str | None = None         # e.g. "biology", "chemistry"; informational only
    notes: str | None = None
    # How many hypotheses this run should generate. Derived from the
    # discussion by ParseGoal (default 15), overridable via --n.
    n_ideas: int = 15


class Session(BaseModel):
    id: str
    created_at: datetime
    updated_at: datetime
    status: SessionStatus
    research_goal: str
    research_plan: ResearchPlan
    config_snapshot: dict
    budget_tokens: int
    budget_usd: float
    budget_used_tokens: int = 0
    budget_used_usd: float = 0.0
    wall_deadline: datetime | None = None
    final_overview: str | None = None     # relative path under data_dir
