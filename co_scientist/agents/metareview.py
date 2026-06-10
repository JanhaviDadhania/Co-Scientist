"""Meta-review agent — periodic system feedback + final research overview.

Two actions:
- `GenerateSystemFeedback`           — Sonnet + thinking; writes a SystemFeedback row.
  The body is auto-injected into future Generation/Evolution prompts via the
  `latest_system_feedback` query the agents already perform.
- `GenerateFinalResearchOverview`    — Opus + max thinking; writes the markdown
  report and updates `sessions.final_overview`.
"""

from __future__ import annotations

from datetime import UTC, datetime

from .. import ids
from ..llm.anthropic_client import AgentCallSpec, CachedBlock, CallContext
from ..llm.prompts import render
from ..llm.routing import route
from ..logging import get_logger
from ..models import SystemFeedback, Task, TaskResult
from ..storage.artifacts import write_json, write_text
from ..storage.repos import feedback as fb_repo
from ..storage.repos import hypotheses as hyp_repo
from ..storage.repos import reviews as rev_repo
from ..storage.repos import sessions as sess_repo
from ..storage.repos import tournaments as tourney_repo
from .base import BaseAgent
from .schemas import RECORD_SYSTEM_FEEDBACK_TOOL

log = get_logger("metareview")


class MetaReviewAgent(BaseAgent):
    name = "metareview"

    async def execute(self, task: Task) -> TaskResult:
        if task.action == "GenerateSystemFeedback":
            return await self._system_feedback(task)
        if task.action == "GenerateFinalResearchOverview":
            return await self._final_overview(task)
        raise ValueError(f"MetaReviewAgent does not handle action {task.action!r}")

    # ----------------------------- system feedback ----------------------------- #

    async def _system_feedback(self, task: Task) -> TaskResult:
        session = await sess_repo.fetch(self.deps.db, task.session_id)
        if session is None:
            raise RuntimeError(f"session {task.session_id} missing")

        reviews = await rev_repo.list_for_session(self.deps.db, session.id)
        if not reviews:
            return TaskResult(kind="noop", extra={"reason": "no reviews yet"})

        reviews_block = "\n\n---\n\n".join(
            f"### Review of `{r.hypothesis_id}` (kind={r.kind}, verdict={r.verdict or '?'})\n{r.body[:3000]}"
            for r in reviews[:50]
        )
        rationales = await tourney_repo.recent_rationales(self.deps.db, session.id, limit=50)
        debate_block = "\n\n---\n\n".join(rat[:1500] for rat in rationales if rat)

        prompt = render(
            "metareview.system",
            goal=session.research_plan.objective,
            preferences="; ".join(session.research_plan.preferences),
            reviews=reviews_block,
            debate_rationales=debate_block,
        )
        r = route(self.deps.cfg, "metareview", "system")
        sys_blocks = [
            CachedBlock(self._system_prompt_header(), cache=True),
            CachedBlock(
                f"# Research goal\n{session.research_goal}\n\n"
                f"# Preferences\n{'; '.join(session.research_plan.preferences)}",
                cache=True,
            ),
        ]
        survey = self._field_context_block()
        if survey is not None:
            sys_blocks.append(survey)
        spec = AgentCallSpec(
            route=r,
            system_blocks=sys_blocks,
            user_blocks=[CachedBlock(prompt, cache=False)],
            tools=[RECORD_SYSTEM_FEEDBACK_TOOL],
            tool_choice={"type": "tool", "name": "record_system_feedback"},
            max_output_tokens=4096,
        )
        ctx = CallContext(
            session_id=session.id, task_id=task.id,
            agent="metareview", action="GenerateSystemFeedback", mode="system",
        )
        resp = await self.deps.llm.call(spec, ctx)
        record = self._final_tool_use(resp, "record_system_feedback")
        if record is None:
            return TaskResult(kind="noop", extra={"reason": "no record_system_feedback"})

        narrative = record.get("narrative") or ""
        if record.get("common_weaknesses"):
            narrative += "\n\n**Common weaknesses:** " + "; ".join(record["common_weaknesses"])
        if record.get("common_strengths"):
            narrative += "\n\n**Common strengths:** " + "; ".join(record["common_strengths"])
        if record.get("suggested_focus_areas"):
            narrative += "\n\n**Suggested focus:** " + "; ".join(record["suggested_focus_areas"])

        fb_id = ids.feedback_id()
        artifact_path = await write_json(
            self.deps.cfg, session.id, "system_feedback", fb_id, record
        )
        await fb_repo.insert(self.deps.db, SystemFeedback(
            id=fb_id, session_id=session.id, created_at=datetime.now(UTC),
            source="meta_review", kind="system_feedback",
            target_id=None, text=narrative.strip()[:8000],
            artifact_path=artifact_path, active=True,
        ))
        return TaskResult(
            kind="system_feedback_generated",
            extra={"feedback_id": fb_id, "n_reviews": len(reviews)},
        )

    # ----------------------------- final overview ----------------------------- #

    async def _final_overview(self, task: Task) -> TaskResult:
        session = await sess_repo.fetch(self.deps.db, task.session_id)
        if session is None:
            raise RuntimeError(f"session {task.session_id} missing")

        # ALL hypotheses — no top-K cap. The deliverable the scientist reads
        # must not be synthesized from a truncated view of the run.
        all_hyps = await hyp_repo.list_for_session(self.deps.db, session.id)
        if not all_hyps:
            return TaskResult(kind="noop", extra={"reason": "no hypotheses"})
        hyps = sorted(all_hyps, key=lambda h: -(h.elo if h.elo is not None else -1))

        # Fetch all reviews for the session in one query, then group by
        # hypothesis_id. Beats N+1 list_for_hypothesis() calls.
        reviews_by_hyp: dict[str, list] = {}
        for rv in await rev_repo.list_for_session(self.deps.db, session.id):
            reviews_by_hyp.setdefault(rv.hypothesis_id, []).append(rv)

        # Per hypothesis: FULL text (not the truncated summary), the best
        # review's full body, and the winning debate rationale — everything
        # the metareview.final prompt promises it was given.
        chunks: list[str] = []
        for h in hyps:
            rs = reviews_by_hyp.get(h.id, [])
            best_review = None
            if rs:
                rs_sorted = sorted(
                    rs, key=lambda r: (r.kind != "full", -(r.scores.novelty or 0))
                )
                best_review = rs_sorted[0].body
            rationale = await tourney_repo.winning_rationale_for(
                self.deps.db, session.id, h.id
            )
            elo_s = f"{h.elo:.0f}" if h.elo is not None else "—"
            chunk = (
                f"### `{h.id}` (Elo {elo_s}, strategy `{h.strategy}`, "
                f"state `{h.state}`, matches {h.matches_played})\n\n"
                f"{h.full_text}\n\n"
                f"**Best review:**\n{best_review or '(none)'}"
            )
            if rationale:
                chunk += f"\n\n**Winning debate rationale:**\n{rationale}"
            chunks.append(chunk)
        top_block = "\n\n---\n\n".join(chunks)

        composed_fb = await fb_repo.composed_feedback(self.deps.db, session.id)

        prompt = render(
            "metareview.final",
            goal=session.research_plan.objective,
            preferences="; ".join(session.research_plan.preferences),
            system_feedback=composed_fb or "",
            top_hypotheses_block=top_block,
        )
        r = route(self.deps.cfg, "metareview", "final")
        sys_blocks = [
            CachedBlock(self._system_prompt_header(), cache=True),
            CachedBlock(
                f"# Research goal\n{session.research_goal}\n\n"
                f"# Preferences\n{'; '.join(session.research_plan.preferences)}",
                cache=True,
            ),
        ]
        survey = self._field_context_block()
        if survey is not None:
            sys_blocks.append(survey)
        spec = AgentCallSpec(
            route=r,
            system_blocks=sys_blocks,
            user_blocks=[CachedBlock(prompt, cache=False)],
            tools=[],            # No tools — write the markdown directly
            tool_choice=None,
            max_output_tokens=8192,
        )
        ctx = CallContext(
            session_id=session.id, task_id=task.id,
            agent="metareview", action="GenerateFinalResearchOverview", mode="final",
        )
        resp = await self.deps.llm.call(spec, ctx)
        text = self._final_text(resp)
        if not text.strip():
            text = "# Research overview\n\n_(No content was generated; see transcripts.)_"

        overview_path = await write_text(
            self.deps.cfg, session.id, "final", "overview", ".md", text
        )
        return TaskResult(
            kind="final_overview_generated",
            extra={"overview_path": overview_path, "n_hypotheses": len(hyps)},
        )
