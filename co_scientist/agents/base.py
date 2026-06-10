"""BaseAgent — shared run-loop plumbing for all six specialized agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import aiosqlite

from ..config import Config
from ..llm.anthropic_client import CachedBlock
from ..llm.provider import LLMProvider
from ..models import Task, TaskResult
from ..safety.quoting import SAFETY_PREAMBLE
from ..tools.registry import ToolRegistry


@dataclass
class AgentDeps:
    """Bundle of resources every agent needs."""

    cfg: Config
    db: aiosqlite.Connection
    llm: LLMProvider
    tools: ToolRegistry


class BaseAgent:
    name: str = "base"

    def __init__(self, deps: AgentDeps) -> None:
        self.deps = deps

    # Subclasses override
    async def execute(self, task: Task) -> TaskResult:  # pragma: no cover
        raise NotImplementedError

    # ----------------------------- helpers ----------------------------- #

    def _system_prompt_header(self) -> str:
        """Common safety preamble prepended to every agent's system prompt."""
        return (
            f"You are the {self.name} agent in a multi-agent scientific research system. "
            f"Operate carefully and cite your sources. {SAFETY_PREAMBLE}"
        )

    def _field_context_block(self, *, tools_note: str | None = None) -> CachedBlock | None:
        """The curated field survey as a cacheable context block.

        The same block (same text → same cache entry) travels to every agent
        so the survey grounds Generation, Reflection, Ranking, Evolution, and
        Meta-review alike — not just the first hop.
        """
        fc = (self.deps.cfg.run.field_context or "").strip()
        if not fc:
            return None
        extra = f"\n\n{tools_note}" if tools_note else ""
        return CachedBlock(
            "# Field background — required reading\n\n"
            "A curated survey of the field this session investigates is "
            "provided below. Ground your reasoning and judgments in it; if it "
            f"covers a topic, trust it and cite it.{extra}\n\n"
            "--- BEGIN SURVEY ---\n"
            f"{fc}\n"
            "--- END SURVEY ---",
            cache=True,
        )

    async def _record_discard(
        self,
        session_id: str,
        hypothesis_id: str,
        record: dict[str, Any],
        *,
        strategy: str,
        duplicate_of: str,
        similarity: float | None,
    ) -> None:
        """Make a dedup rejection visible: discarded/ artifact + event.

        Nothing the LLM generates is ever lost silently — the full record
        lands in discarded/ with the duplicate-of id and similarity, and an
        event marks that it happened.
        """
        from ..storage.artifacts import write_json
        from ..storage.repos import events as events_repo

        path: str | None = None
        try:
            path = await write_json(
                self.deps.cfg, session_id, "discarded", hypothesis_id,
                {
                    "reason": "near_duplicate",
                    "duplicate_of": duplicate_of,
                    "similarity": similarity,
                    "strategy": strategy,
                    "record": record,
                },
            )
        except Exception:  # noqa: BLE001 — the event below still fires
            pass
        await events_repo.emit(
            self.deps.db, session_id=session_id, task_id=None, agent=self.name,
            event="hypothesis_deduped",
            payload={
                "hypothesis_id": hypothesis_id,
                "duplicate_of": duplicate_of,
                "similarity": similarity,
                "artifact_path": path,
            },
        )

    @staticmethod
    def _final_tool_use(response, tool_name: str) -> dict[str, Any] | None:
        """Find the most recent tool_use block with the given name in a response.

        Returns the .input dict, or None if not present.
        """
        for block in reversed(response.raw.content or []):
            if getattr(block, "type", None) == "tool_use" and getattr(block, "name", "") == tool_name:
                inp = getattr(block, "input", None)
                return dict(inp) if isinstance(inp, dict) else None
        return None

    @staticmethod
    def _final_text(response) -> str:
        parts = []
        for block in response.raw.content or []:
            if getattr(block, "type", None) == "text":
                parts.append(getattr(block, "text", ""))
        return "\n".join(parts).strip()
