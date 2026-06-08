"""Generation agent — proposes new hypotheses.

M3 ships the `literature` strategy. `debate` / `assumption` / `feedback_driven`
hook into the same machinery and land in M5+.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

import numpy as np

from .. import ids
from ..llm.anthropic_client import AgentCallSpec, CachedBlock, CallContext
from ..llm.prompts import render
from ..llm.routing import route
from ..llm.tool_loop import ToolLoopExhausted, run_tool_loop
from ..logging import get_logger
from ..models import CitedPaper, Hypothesis, ResearchPlan, Task, TaskResult
from ..safety.quoting import quote_untrusted
from ..storage.artifacts import write_json
from ..storage.repos import embeddings as emb_repo
from ..storage.repos import feedback as fb_repo
from ..storage.repos import hypotheses as hyp_repo
from ..storage.repos import sessions as sess_repo
from ..vectors.embedder import make_embedder
from ..vectors.store import FaissStore
from .base import AgentDeps, BaseAgent
from .schemas import RECORD_HYPOTHESIS_TOOL

log = get_logger("generation")


class GenerationAgent(BaseAgent):
    name = "generation"

    async def execute(self, task: Task) -> TaskResult:
        strategy = task.payload.get("strategy", "literature")
        n_target = int(task.payload.get("n", 3))

        session = await sess_repo.fetch(self.deps.db, task.session_id)
        if session is None:
            raise RuntimeError(f"session {task.session_id} missing")
        plan = session.research_plan

        if strategy != "literature":
            # M3 ships only the literature strategy.
            raise NotImplementedError(f"strategy {strategy!r} lands in a later milestone")

        # 1. Render the prompt and run the tool loop with `record_hypothesis` available.
        articles_block = (
            "You will gather literature using the available tools (web_search, "
            "pubmed_search, arxiv_search, europe_pmc_search, web_fetch). Pull "
            "abstracts for the most relevant items, then synthesize. After you "
            "have surveyed the literature, call `record_hypothesis` exactly once "
            "with your proposed hypothesis.\n\n"
            "IMPORTANT — interpreting empty search results: an empty result set "
            "(no hits) is positive evidence that the literature you searched for "
            "does not exist. When the goal requires a candidate with NO prior "
            "published evidence, empty searches CONFIRM novelty — they are a "
            "reason to PROCEED, not to keep searching. Do not chase confirmation "
            "you will never get. After at most 2-3 searches that return no "
            "relevant hits for a candidate, treat its novelty as established and "
            "call `record_hypothesis`. A recorded hypothesis backed by a few "
            "empty searches is far better than running out of turns with nothing."
        )

        instructions_block = (
            "=========================================================\n"
            "HARD CONSTRAINT — DOWNSTREAM PIPELINE READS YOUR OUTPUT\n"
            "=========================================================\n\n"
            "You are the hypothesis-generation step in an automated multi-agent "
            "research pipeline. Downstream agents (Reflection, Ranking, Evolution, "
            "Meta-review) consume the structured `record_hypothesis` payload you "
            "emit. They do not read your prose, your thinking, or your "
            "intermediate tool-call results — they ONLY read the fields of your "
            "`record_hypothesis` call. The schema and the field set are a "
            "CONTRACT, not a suggestion.\n\n"
            "If you fail to call `record_hypothesis`, or call it with a malformed "
            "payload, or call it with missing required fields, the pipeline halts "
            "and ALL the work you just did — every search, every analysis — is "
            "discarded. There is no human in the loop, no retry, no salvage path.\n\n"
            "WORKFLOW:\n"
            "1. Search the literature as much as you need — no fixed search "
            "budget. Use pubmed_search, arxiv_search, europe_pmc_search, "
            "web_search, web_fetch freely until you feel grounded in the goal.\n"
            "2. Once grounded, call `record_hypothesis` EXACTLY ONCE with the "
            "complete payload specified below. Do not call it twice. Do not "
            "summarize in prose instead of calling it. Do not ask clarifying "
            "questions. Do not describe what you would do — DO IT.\n\n"
            "EXACT FORMAT — every field below is required by downstream agents:\n"
            "  • title — short noun-phrase title (string, non-empty)\n"
            "  • statement — ONE sentence stating the hypothesis (string, non-empty)\n"
            "  • mechanism — detailed causal/mechanistic story (string, "
            "multi-paragraph OK, but a single coherent string — not an array)\n"
            "  • entities — array of specific named actors (proteins, materials, "
            "datasets, models, etc.) as strings; never an array of objects\n"
            "  • anticipated_outcomes — concrete observations that would confirm "
            "the hypothesis if true (string)\n"
            "  • novelty_argument — what is new relative to the cited literature "
            "(string)\n"
            "  • citations — array of objects, each with at minimum {url, title, "
            "excerpt}. Every url MUST be a url you actually opened during this "
            "task's tool calls. If you did not open any urls, return [] — DO NOT "
            "fabricate urls.\n\n"
            "Propose EXACTLY ONE hypothesis — the strongest you can justify. "
            "Additional hypotheses come from separate Generation calls.\n\n"
            "FINAL REMINDER: your ONLY valid exit from this task is a single "
            "`record_hypothesis` call whose payload matches the exact field set "
            "above. Anything else breaks the pipeline."
        )
        prompt = render(
            "generation.literature",
            goal=plan.objective,
            preferences="; ".join(plan.preferences),
            articles_with_reasoning=articles_block,
            instructions=instructions_block,
        )
        _ = n_target  # n_target controls how many parallel Generation tasks are enqueued, not per-call output

        sys_blocks = [
            CachedBlock(self._system_prompt_header(), cache=True),
            CachedBlock(
                _build_session_context(session.research_goal, plan,
                                       await _latest_system_feedback(self.deps, session.id)),
                cache=True,
            ),
        ]
        user_blocks: list[CachedBlock] = []
        field_context = (self.deps.cfg.run.field_context or "").strip()
        if field_context:
            user_blocks.append(CachedBlock(
                "# Field background — required reading\n\n"
                "A curated survey of the field this hypothesis must address has "
                "been provided below. READ IT IN FULL before doing anything else. "
                "Your hypothesis MUST be informed by and consistent with this "
                "material.\n\n"
                "The literature search tools (pubmed_search, arxiv_search, "
                "europe_pmc_search, web_search, web_fetch) are available but "
                "should only be used to fill specific gaps the survey leaves "
                "open — NOT to duplicate searches the survey has already done. "
                "If the survey covers a topic, trust it and cite it; do not "
                "re-search. If the survey is silent on a specific point your "
                "hypothesis depends on, THEN you may search for that specific "
                "point.\n\n"
                "--- BEGIN SURVEY ---\n"
                f"{field_context}\n"
                "--- END SURVEY ---",
                cache=True,
            ))
        user_blocks.append(CachedBlock(prompt, cache=False))

        r = route(self.deps.cfg, "generation", "literature")
        tools = [*self.deps.tools.anthropic_tools_for("generation"), RECORD_HYPOTHESIS_TOOL]

        spec = AgentCallSpec(
            route=r,
            system_blocks=sys_blocks,
            user_blocks=user_blocks,
            tools=tools,
            tool_choice={"type": "auto"},
            # A full record_hypothesis payload (statement + mechanism + entities
            # + outcomes + novelty + citations) is large; verbose / reasoning
            # models overran the old 4096 cap mid-JSON, so the arguments string
            # was truncated and unparseable. 8192 leaves room to complete it.
            max_output_tokens=8192,
        )
        ctx = CallContext(
            session_id=task.session_id, task_id=task.id,
            agent="generation", action="CreateInitialHypotheses", mode="literature",
        )

        try:
            loop_result = await run_tool_loop(
                self.deps.llm,
                spec=spec, ctx=ctx,
                registry=self.deps.tools,
                max_iters=self.deps.cfg.tool_loop.generation_max_iters,
                parallel_cap=self.deps.cfg.tool_loop.parallel_cap,
                tool_timeout_s=self.deps.cfg.tool_loop.tool_timeout_seconds,
                force_terminal_tool="record_hypothesis",
            )
        except ToolLoopExhausted as e:
            raise RuntimeError(f"generation exhausted tool loop: {e}") from e

        # 2. Extract record_hypothesis from the final assistant message.
        record = self._final_tool_use(loop_result.response, "record_hypothesis")
        if record is None:
            raise RuntimeError("Generation did not call record_hypothesis")

        # 3. Validate every citation URL is in the union of URLs seen during the loop.
        record["citations"] = _filter_to_seen_urls(record.get("citations", []), loop_result.seen_urls)

        # 4. Persist + embed + dedup-check.
        hid, was_new = await self._persist(session.id, record, strategy="literature")
        return TaskResult(
            kind="hypothesis_created",
            hypothesis_ids=[hid] if was_new else [],
            extra={"tool_calls": loop_result.tool_calls, "iterations": loop_result.iterations},
        )

    # ---------------------------------------------------------------- #

    async def _persist(
        self, session_id: str, record: dict[str, Any], *, strategy: str
    ) -> tuple[str, bool]:
        statement = record.get("statement") or record.get("title") or ""
        if not statement:
            raise ValueError("record_hypothesis: missing statement")

        origin = f"generation/{strategy}"
        hid = ids.hypothesis_id(session_id, origin, statement)
        summary = (record.get("statement") or "") + "\n\n" + (record.get("mechanism") or "")
        full_text = _render_hypothesis_md(record)

        # Write the JSON artifact first so the row points at a real file.
        artifact_path = await write_json(
            self.deps.cfg, session_id, "hypotheses", hid,
            {"strategy": strategy, "record": record},
        )

        citations = [
            CitedPaper(
                title=c.get("title", ""),
                url=c.get("url", ""),
                excerpt=c.get("excerpt"),
                doi=c.get("doi"),
                year=c.get("year"),
            )
            for c in record.get("citations", [])
            if isinstance(c, dict) and c.get("url")
        ]

        # Step 1: embed + near-neighbour check (does NOT mutate FAISS).
        try:
            dup_id, embed_payload = await self._dedup_query(session_id, summary)
        except Exception as e:
            log.warning("dedup_query_failed", err=str(e))
            dup_id, embed_payload = None, None

        if dup_id is not None and dup_id != hid:
            # Found a near-duplicate already in this session: skip insert + skip FAISS.
            return dup_id, False

        # Step 2: insert the hypothesis row. Deterministic IDs make this idempotent.
        h = Hypothesis(
            id=hid,
            session_id=session_id,
            created_at=datetime.now(UTC),
            created_by="generation",
            strategy=strategy,        # type: ignore[arg-type]
            parent_ids=record.get("parent_ids") or [],
            title=record.get("title", "")[:300],
            summary=(record.get("statement") or "")[:1000],
            full_text=full_text,
            citations=citations,
            artifact_path=artifact_path,
            state="draft",
        )
        inserted = await hyp_repo.insert(self.deps.db, h)

        # Step 3: only add to FAISS if we actually inserted a new row, so FAISS and
        # the hypotheses table can never disagree (FK in embeddings_meta enforces it).
        if inserted and embed_payload is not None:
            try:
                await self._dedup_commit(session_id, hid, embed_payload)
            except Exception as e:
                log.warning("dedup_commit_failed", hypothesis_id=hid, err=str(e))

        return hid, inserted

    async def _dedup_query(
        self, session_id: str, text: str
    ) -> tuple[str | None, dict[str, Any] | None]:
        """Read-only: embed + nearest-neighbour search. No FAISS mutation.

        Returns (existing_duplicate_id_or_None, embed_payload_for_later_commit).
        """
        try:
            embedder = make_embedder(self.deps.cfg)
        except (RuntimeError, ValueError):
            return None, None
        vec = await embedder.embed([text])
        if vec.size == 0:
            return None, None
        v = vec[0]
        store = FaissStore(self.deps.cfg, session_id, dim=embedder.dim)
        await store.load_or_create()
        nearest = await store.search(np.asarray(v), k=1)
        thr = self.deps.cfg.vectors.dedup_cosine_threshold
        if nearest and nearest[0][1] >= thr:
            return nearest[0][0], None
        payload = {
            "vector": np.asarray(v),
            "model": embedder.model,
            "dim": embedder.dim,
            "text_hash": ids.text_hash(text),
        }
        return None, payload

    async def _dedup_commit(
        self, session_id: str, hypothesis_id: str, payload: dict[str, Any]
    ) -> None:
        """Write-side of dedup: add to FAISS + register the embedding."""
        store = FaissStore(self.deps.cfg, session_id, dim=payload["dim"])
        await store.load_or_create()
        offset = await store.add(hypothesis_id, payload["vector"])
        await store.save()
        await emb_repo.upsert(
            self.deps.db,
            id_=ids.embedding_id(hypothesis_id, payload["model"]),
            session_id=session_id,
            hypothesis_id=hypothesis_id,
            model=payload["model"],
            dim=payload["dim"],
            faiss_offset=offset,
            text_hash=payload["text_hash"],
        )


# --------------------------------------------------------------------------- #
# helpers


def _filter_to_seen_urls(
    citations: list[dict[str, Any]], seen: Iterable[str]
) -> list[dict[str, Any]]:
    seen_set = set(seen)
    return [c for c in citations if isinstance(c, dict) and c.get("url") in seen_set]


def _render_hypothesis_md(record: dict[str, Any]) -> str:
    parts: list[str] = []
    if record.get("title"):
        parts.append(f"# {record['title']}")
    parts.append(f"**Hypothesis.** {record.get('statement', '')}")
    if record.get("mechanism"):
        parts.append(f"## Mechanism\n{record['mechanism']}")
    if record.get("entities"):
        parts.append("## Entities\n- " + "\n- ".join(record["entities"]))
    if record.get("anticipated_outcomes"):
        parts.append(f"## Anticipated outcomes\n{record['anticipated_outcomes']}")
    if record.get("novelty_argument"):
        parts.append(f"## Novelty\n{record['novelty_argument']}")
    if record.get("citations"):
        parts.append("## Citations")
        for c in record["citations"]:
            year = f" ({c.get('year')})" if c.get("year") else ""
            parts.append(f"- {c.get('title','(no title)')}{year} — {c.get('url','')}")
    return "\n\n".join(parts)


def _build_session_context(goal: str, plan: ResearchPlan, sys_feedback_text: str | None) -> str:
    fb = ""
    if sys_feedback_text:
        fb = "\n\n# Researcher / Meta-review Feedback\n" + quote_untrusted(
            sys_feedback_text, id_="system_feedback:latest"
        )
    return (
        f"# Research goal\n{goal}\n\n"
        f"# Parsed plan\n"
        f"- Objective: {plan.objective}\n"
        f"- Preferences: {'; '.join(plan.preferences) or '(none)'}\n"
        f"- Idea attributes: {'; '.join(plan.idea_attributes) or '(none)'}\n"
        f"- Constraints: {'; '.join(plan.constraints) or '(none)'}\n"
        f"{fb}"
    )


async def _latest_system_feedback(deps: AgentDeps, session_id: str) -> str | None:
    fb = await fb_repo.latest_system_feedback(deps.db, session_id)
    return fb.text if fb is not None else None
