"""Supervisor — durable task scheduler for the multi-agent system.

Responsibilities:
1. Parse the scientist's goal into a ResearchPlan.
2. Bootstrap the session (insert row, reclaim expired leases on resume).
3. Run a bounded asyncio worker pool that claims tasks from the DB-backed queue.
4. Apply follow-up scheduling rules after each task completes.
5. Periodically run `decide_next_steps` when the queue is idle:
   - Tournament refinement.
   - Evolution if the leaderboard is stable.
   - Periodic system-feedback meta-reviews.
6. Check the termination predicate after every task; on stop, cancel pending
   work and run a single final meta-review for the overview.
7. Honor pause / abort via DB-flagged session.status.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime
from typing import Any

import aiosqlite

from .. import ids
from ..config import Config
from ..llm.anthropic_client import (
    AgentCallSpec,
    CachedBlock,
    CallContext,
)
from ..llm.budgets import TokenBudget
from ..llm.prompts import render
from ..llm.provider import get_provider
from ..llm.routing import route
from ..logging import bind, get_logger
from ..models import ResearchPlan, Session, Task
from ..orchestrator.events import GLOBAL_BUS
from ..orchestrator.termination import (
    StabilityTracker,
    StopReason,
    should_stop,
    snapshot_top_k,
)
from ..storage import db as db_mod
from ..storage.artifacts import write_text
from ..storage.repos import events as events_repo
from ..storage.repos import feedback as fb_repo
from ..storage.repos import hypotheses as hyp_repo
from ..storage.repos import reviews as rev_repo
from ..storage.repos import sessions as sess_repo
from ..storage.repos import tasks as task_repo
from ..tools.registry import ToolRegistry
from .base import AgentDeps
from .generation import GenerationAgent
from .ranking import RankingAgent
from .reflection import ReflectionAgent
from .schemas import RECORD_RESEARCH_PLAN_TOOL

log = get_logger("supervisor")

# Default ideas per run when the discussion doesn't say otherwise.
DEFAULT_N_IDEAS = 15


def next_batch_size(remaining: int) -> int:
    """Generation batch sizing: half the remaining target, floor of 3.

    For N=15 this yields 8 → 4 → 3 — a first batch big enough to build a
    real leaderboard, then smaller batches generated AFTER meta-review
    feedback exists, so they are steered by it.
    """
    if remaining <= 0:
        return 0
    if remaining <= 4:
        return remaining
    return -(-remaining // 2)  # ceil(remaining / 2)


def evolution_gate(population: int) -> int:
    """Mature-hypothesis threshold for Evolution, proportional and capped.

    pop 8 → 4 · pop 15 → 8 · pop 30+ → 10, never more.
    """
    return max(4, min(10, -(-population // 2)))


def metareview_due(match_count: int, feedback_count: int, population: int) -> bool:
    """Proportional meta-review trigger: every ~population × 4 matches."""
    if population < 2:
        return False
    interval = max(8, population * 4)
    return match_count >= (feedback_count + 1) * interval


# ----------------------------- public API ----------------------------- #


class Supervisor:
    """One-process Supervisor; CLI invokes via `await supervisor.run_session(...)`."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    async def run_session(
        self,
        goal: str,
        *,
        preferences_text: str | None = None,
        n_initial: int | None = None,
        wall_clock_seconds: int | None = None,
        resume_session_id: str | None = None,
    ) -> str:
        conn = await db_mod.connect(self.cfg)
        try:
            if resume_session_id is None:
                session = await self._create_session(conn, goal, preferences_text, wall_clock_seconds)
                bind(session_id=session.id)
                log.info(
                    "session_started",
                    goal=goal[:120], session_id=session.id,
                    budget_usd=session.budget_usd, n_initial=n_initial,
                )
                await self._emit(conn, session.id, "session_started", {
                    "goal": goal[:200], "n_initial": n_initial,
                    "budget_usd": session.budget_usd,
                })
                budget = TokenBudget(
                    cfg=self.cfg,
                    budget_tokens=session.budget_tokens,
                    budget_usd=session.budget_usd,
                )
                llm = get_provider(self.cfg, db=conn, budget=budget)
                tools = ToolRegistry(self.cfg).discover()
                deps = AgentDeps(cfg=self.cfg, db=conn, llm=llm, tools=tools)

                try:
                    plan = await self._parse_goal(deps, session, goal, preferences_text)
                except Exception:
                    await sess_repo.set_status(conn, session.id, "failed")
                    raise
                # An explicit --n overrides whatever ParseGoal derived.
                if n_initial is not None and n_initial >= 1:
                    plan.n_ideas = min(n_initial, self.cfg.run.max_ideas)
                await self._apply_plan(conn, session, plan)
                session = await sess_repo.fetch(conn, session.id)
                assert session is not None

                # Enqueue only the FIRST batch — the rest arrive via the
                # batched refill in _decide_next_steps, after meta-review
                # feedback exists to steer them.
                first_batch = next_batch_size(plan.n_ideas)
                log.info("generation_batched", n_ideas=plan.n_ideas, first_batch=first_batch)
                for i in range(first_batch):
                    await task_repo.enqueue(conn, Task(
                        id=ids.task_id(), session_id=session.id,
                        created_at=datetime.now(UTC),
                        agent="generation", action="CreateInitialHypotheses",
                        payload={"strategy": "literature", "n": 1},
                        priority=100, status="pending",
                        idempotency_key=f"{session.id}::generation::initial::{i}",
                    ))
            else:
                session = await sess_repo.fetch(conn, resume_session_id)
                if session is None:
                    raise RuntimeError(f"no such session: {resume_session_id}")
                bind(session_id=session.id)
                log.info("session_resumed", session_id=session.id, status=session.status)
                # Restore run inputs that live only in the session's config
                # snapshot (the resume CLI path has no --field-context /
                # --discussion flags) so the survey and discussion are never
                # silently dropped on resume.
                snap_run = (session.config_snapshot or {}).get("run") or {}
                if not self.cfg.run.field_context and snap_run.get("field_context"):
                    self.cfg.run.field_context = snap_run["field_context"]
                    log.info("field_context_restored",
                             chars=len(self.cfg.run.field_context))
                if not self.cfg.run.discussion and snap_run.get("discussion"):
                    self.cfg.run.discussion = snap_run["discussion"]
                    log.info("discussion_restored",
                             chars=len(self.cfg.run.discussion))
                reclaimed = await task_repo.reclaim_expired_leases(
                    conn, session.id, max_attempts=self.cfg.lease.max_attempts,
                )
                log.info("leases_reclaimed", **reclaimed)
                if session.status not in ("running", "paused"):
                    await sess_repo.set_status(conn, session.id, "running")
                budget = TokenBudget(
                    cfg=self.cfg,
                    budget_tokens=session.budget_tokens,
                    budget_usd=session.budget_usd,
                )
                llm = get_provider(self.cfg, db=conn, budget=budget)
                tools = ToolRegistry(self.cfg).discover()
                deps = AgentDeps(cfg=self.cfg, db=conn, llm=llm, tools=tools)

            tracker = StabilityTracker(
                k=self.cfg.termination.elo_stability_k,
                n=self.cfg.termination.elo_stability_n,
                eps=self.cfg.termination.elo_stability_eps,
            )

            stop_reason = await self._main_loop(conn, deps, session, tracker)
            log.info("main_loop_exit", stop_reason=stop_reason.value if stop_reason else "none")

            await self._finalize(conn, deps, session, stop_reason)
            return session.id
        finally:
            await conn.close()

    # ----------------------------- session bootstrap ----------------------------- #

    async def _create_session(
        self,
        conn: aiosqlite.Connection,
        goal: str,
        preferences_text: str | None,
        wall_clock_seconds: int | None,
    ) -> Session:
        sid = ids.session_id()
        now = datetime.now(UTC)
        wall = wall_clock_seconds or self.cfg.run.wall_clock_seconds
        from datetime import timedelta

        plan = ResearchPlan(objective=goal.strip(), preferences=[], idea_attributes=[])
        snap: dict[str, Any] = json.loads(json.dumps(self.cfg.model_dump(exclude={"secrets"})))
        s = Session(
            id=sid, created_at=now, updated_at=now, status="running",
            research_goal=goal, research_plan=plan,
            config_snapshot=snap,
            budget_tokens=self.cfg.run.budget_tokens, budget_usd=self.cfg.run.budget_usd,
            wall_deadline=now + timedelta(seconds=wall),
        )
        await sess_repo.insert(conn, s)
        if preferences_text:
            await fb_repo.insert(conn, _human_preference(s.id, preferences_text))
        return s

    async def _parse_goal(
        self,
        deps: AgentDeps,
        session: Session,
        goal: str,
        preferences_text: str | None,
    ) -> ResearchPlan:
        prompt = render(
            "parse_goal", goal=goal,
            preferences_text=preferences_text or "",
            discussion=(self.cfg.run.discussion or "").strip(),
        )
        r = route(self.cfg, "parse_goal", None)
        spec = AgentCallSpec(
            route=r,
            system_blocks=[CachedBlock("You parse research goals into structured plans.", cache=True)],
            user_blocks=[CachedBlock(prompt, cache=False)],
            tools=[RECORD_RESEARCH_PLAN_TOOL],
            tool_choice={"type": "tool", "name": "record_research_plan"},
            max_output_tokens=1024,
        )
        ctx = CallContext(
            session_id=session.id, task_id=None,
            agent="parse_goal", action="parse_goal", mode=None,
        )
        resp = await deps.llm.call(spec, ctx)
        record: dict[str, Any] | None = None
        for b in resp.raw.content:
            if getattr(b, "type", None) == "tool_use" and getattr(b, "name", "") == "record_research_plan":
                inp = getattr(b, "input", None)
                if isinstance(inp, dict):
                    record = inp
                    break
        if record is None:
            # No silent fallback: a degraded bare plan would scope every later
            # prompt with empty preferences/constraints for the whole session.
            # The provider already validated + retried; if we still have no
            # record, halt loudly.
            raise RuntimeError(
                "parse_goal did not produce a record_research_plan payload "
                "after validation retries — halting (no bare-plan fallback)"
            )
        n_ideas = record.get("n_ideas")
        if not isinstance(n_ideas, int) or n_ideas < 1:
            n_ideas = DEFAULT_N_IDEAS
        n_ideas = min(n_ideas, self.cfg.run.max_ideas)
        return ResearchPlan(
            objective=record.get("objective", goal.strip()),
            preferences=record.get("preferences", []),
            constraints=record.get("constraints", []),
            idea_attributes=record.get("idea_attributes", []),
            domain_hint=record.get("domain_hint") or None,
            notes=record.get("notes") or None,
            n_ideas=n_ideas,
        )

    async def _apply_plan(
        self, conn: aiosqlite.Connection, session: Session, plan: ResearchPlan
    ) -> None:
        await conn.execute(
            "UPDATE sessions SET research_plan=?, updated_at=? WHERE id=?",
            (plan.model_dump_json(), datetime.now(UTC).isoformat(), session.id),
        )
        await conn.commit()

    # ----------------------------- main loop ----------------------------- #

    async def _main_loop(
        self,
        conn: aiosqlite.Connection,
        deps: AgentDeps,
        session: Session,
        tracker: StabilityTracker,
    ) -> StopReason | None:
        agents = self._build_agents(deps)
        sem = asyncio.Semaphore(self.cfg.run.concurrency)
        inflight: set[asyncio.Task] = set()
        worker_seq = 0
        last_decide_at = 0.0
        last_snapshot_match_count = -1

        async def _run_task(t: Task) -> None:
            bind(session_id=session.id, task_id=t.id, agent=t.agent)
            async with sem:
                await task_repo.mark_in_progress(conn, t.id)
                await self._emit(conn, session.id, "task_started",
                                 {"task_id": t.id, "agent": t.agent, "action": t.action,
                                  "target": t.target_id})
                agent = agents.get(t.agent)
                if agent is None:
                    await task_repo.fail(conn, t.id, error=f"no agent: {t.agent}",
                                          max_attempts=self.cfg.lease.max_attempts)
                    return
                try:
                    result = await agent.execute(t)
                except Exception as e:
                    await task_repo.fail(conn, t.id, error=str(e),
                                          max_attempts=self.cfg.lease.max_attempts)
                    log.exception("task_failed", err=str(e), task_id=t.id, action=t.action)
                    await self._emit(conn, session.id, "task_failed",
                                     {"task_id": t.id, "err": str(e)[:300]})
                    return

                await self._apply_follow_ups(conn, session, t, result)
                await task_repo.complete(conn, t.id)
                await self._emit(conn, session.id, "task_completed",
                                 {"task_id": t.id, "kind": result.kind,
                                  "follow_hypothesis_ids": result.hypothesis_ids[:5]})

        try:
            while True:
                # Check external pause/abort by re-reading session status.
                refreshed = await sess_repo.fetch(conn, session.id)
                external_stop = refreshed is not None and refreshed.status in ("aborted",)
                if refreshed is not None and refreshed.status == "paused":
                    # Wait until unpaused (or aborted).
                    await asyncio.sleep(1.0)
                    continue

                # Termination check (refreshes budget_used_* from the row)
                if refreshed is not None:
                    stop = should_stop(self.cfg, refreshed, tracker, external_stop=external_stop)
                    if stop is not None:
                        # Wait for inflight to drain before returning.
                        if inflight:
                            await asyncio.wait(inflight)
                        return stop

                # Refill worker slots.
                slots_open = self.cfg.run.concurrency - len(inflight)
                claimed: list[Task] = []
                for _ in range(slots_open):
                    t = await task_repo.claim_one(
                        conn, session.id, worker_id=f"w{worker_seq}",
                        lease_seconds=self.cfg.lease.default_seconds,
                    )
                    if t is None:
                        break
                    worker_seq += 1
                    claimed.append(t)
                for t in claimed:
                    inflight.add(asyncio.create_task(_run_task(t)))

                # Update stability snapshot when match count crossed the threshold.
                snap = await snapshot_top_k(conn, session.id, self.cfg.termination.elo_stability_k)
                if (
                    snap.match_count >= last_snapshot_match_count + self.cfg.termination.match_snapshot_every
                ):
                    tracker.push(snap)
                    last_snapshot_match_count = snap.match_count
                    log.info(
                        "elo_snapshot", match_count=snap.match_count,
                        top_ids=list(snap.top_ids), top_elos=list(snap.top_elos),
                    )

                # If nothing to do at all and the queue is empty, run decide_next_steps
                # at most every ~10s, else exit (only if we have no hypotheses yet either).
                if not inflight and not claimed:
                    pending = await task_repo.count_by_status(conn, session.id)
                    if pending.get("pending", 0) == 0:
                        now = time.monotonic()
                        if now - last_decide_at >= 10.0:
                            last_decide_at = now
                            scheduled = await self._decide_next_steps(conn, session)
                            if scheduled == 0:
                                # truly idle and no progress possible — exit gracefully
                                return StopReason.IDLE
                            continue
                        # Wait briefly so we don't spin
                        await asyncio.sleep(1.0)
                        continue

                if not inflight:
                    # Nothing claimed AND nothing running — but tasks may be pending
                    # in other workers' future claims; brief sleep and retry.
                    await asyncio.sleep(0.1)
                    continue

                _done, pending = await asyncio.wait(
                    inflight, return_when=asyncio.FIRST_COMPLETED
                )
                inflight = set(pending)
        finally:
            if inflight:
                # Best effort: let any inflight task finish before returning.
                await asyncio.wait(inflight)

    # ----------------------------- follow-up rules ----------------------------- #

    async def _apply_follow_ups(
        self,
        conn: aiosqlite.Connection,
        session: Session,
        task: Task,
        result,
    ) -> None:
        if result.kind == "hypothesis_created":
            for hid in result.hypothesis_ids:
                await task_repo.enqueue(conn, Task(
                    id=ids.task_id(), session_id=session.id,
                    created_at=datetime.now(UTC),
                    agent="reflection", action="ReviewHypothesis",
                    target_id=hid, payload={"kind": "full"},
                    priority=100, status="pending",
                    idempotency_key=f"{hid}::review::full",
                ))
        elif result.kind == "review_completed":
            for hid in result.hypothesis_ids:
                await task_repo.enqueue(conn, Task(
                    id=ids.task_id(), session_id=session.id,
                    created_at=datetime.now(UTC),
                    agent="ranking", action="AddToTournament",
                    target_id=hid, payload={}, priority=80, status="pending",
                    idempotency_key=f"{hid}::ranking::add",
                ))
        elif result.kind == "added_to_tournament":
            for hid in result.hypothesis_ids:
                await task_repo.enqueue(conn, Task(
                    id=ids.task_id(), session_id=session.id,
                    created_at=datetime.now(UTC),
                    agent="ranking", action="RunTournamentBatch",
                    target_id=None,
                    payload={"focus": hid}, priority=120, status="pending",
                    idempotency_key=f"{hid}::ranking::focus_batch",
                ))
        elif result.kind == "tournament_match_complete":
            n_matches = result.extra.get("total_matches_after")
            _ = n_matches
            # Periodically re-cluster the proximity graph.
            from ..storage.repos import tournaments as tourney_repo

            mc = await tourney_repo.count_matches(conn, session.id)
            if (
                mc > 0
                and mc % self.cfg.vectors.full_recluster_every_matches == 0
            ):
                await task_repo.enqueue(conn, Task(
                    id=ids.task_id(), session_id=session.id,
                    created_at=datetime.now(UTC),
                    agent="proximity", action="UpdateProximityGraph",
                    target_id=None, payload={"rebuild": True},
                    priority=200, status="pending",
                    idempotency_key=f"{session.id}::proximity::{mc}",
                ))

    # ----------------------------- decide_next_steps ----------------------------- #

    async def _decide_next_steps(
        self, conn: aiosqlite.Connection, session: Session
    ) -> int:
        """When the queue empties: refill it with refinement work. Returns # enqueued."""
        from ..storage.repos import tournaments as tourney_repo

        enqueued = 0

        # We anchor idle-refinement idempotency keys on the current match count
        # rather than a fresh task id. Otherwise every idle pass — which can
        # fire every ~10s — would enqueue a *new* tournament/evolution task
        # even when a prior one is still pending, flooding the queue and
        # double-counting work toward the budget.
        anchor_mc = await tourney_repo.count_matches(conn, session.id)

        # Batched Generation refill: while fewer generation-born hypotheses
        # exist than the plan's target, enqueue the next batch. This is what
        # makes the meta-review → generation learning loop real — refill
        # batches START after feedback exists, so they pull it in.
        n_target = getattr(session.research_plan, "n_ideas", DEFAULT_N_IDEAS)
        async with conn.execute(
            """SELECT COUNT(*) AS n FROM hypotheses
                  WHERE session_id=? AND created_by='generation'""",
            (session.id,),
        ) as cur:
            row = await cur.fetchone()
        gen_count = row["n"] if row else 0
        batch = next_batch_size(n_target - gen_count)
        if batch > 0:
            for i in range(batch):
                await task_repo.enqueue(conn, Task(
                    id=ids.task_id(), session_id=session.id,
                    created_at=datetime.now(UTC),
                    agent="generation", action="CreateInitialHypotheses",
                    payload={"strategy": "literature", "n": 1},
                    priority=100, status="pending",
                    idempotency_key=f"{session.id}::generation::refill::{gen_count}::{i}",
                ))
            enqueued += batch
            log.info("generation_refill", have=gen_count, target=n_target, batch=batch)

        # Always: one tournament batch to keep refining Elo.
        in_tournament = await hyp_repo.list_for_session(
            conn, session.id, state="in_tournament"
        )
        if len(in_tournament) >= 2:
            await task_repo.enqueue(conn, Task(
                id=ids.task_id(), session_id=session.id,
                created_at=datetime.now(UTC),
                agent="ranking", action="RunTournamentBatch",
                target_id=None, payload={},
                priority=150, status="pending",
                idempotency_key=f"{session.id}::ranking::idle::{anchor_mc}",
            ))
            enqueued += 1

        # Evolve when the leaderboard has matured. The gate is proportional
        # to the population and capped at 10 — the old flat ≥20 was
        # unreachable at realistic population sizes.
        population = len(in_tournament)
        mature = sum(1 for h in in_tournament if h.matches_played >= 3)
        if mature >= evolution_gate(population):
            await task_repo.enqueue(conn, Task(
                id=ids.task_id(), session_id=session.id,
                created_at=datetime.now(UTC),
                agent="evolution", action="EvolveTopHypotheses",
                target_id=None,
                payload={"top_k": 5, "strategies": ["combine", "simplify", "out_of_box"]},
                priority=140, status="pending",
                idempotency_key=f"{session.id}::evolution::idle::{anchor_mc}",
            ))
            enqueued += 1

        # Periodic meta-review — proportional trigger (~population × 4 matches).
        mc = await tourney_repo.count_matches(conn, session.id)
        async with conn.execute(
            """SELECT COUNT(*) AS n FROM system_feedback
                  WHERE session_id=? AND kind='system_feedback' AND source='meta_review'""",
            (session.id,),
        ) as cur:
            row = await cur.fetchone()
        feedback_count = row["n"] if row else 0
        if metareview_due(mc, feedback_count, population):
            await task_repo.enqueue(conn, Task(
                id=ids.task_id(), session_id=session.id,
                created_at=datetime.now(UTC),
                agent="metareview", action="GenerateSystemFeedback",
                target_id=None, payload={},
                priority=180, status="pending",
                idempotency_key=f"{session.id}::metareview::feedback::{feedback_count + 1}",
            ))
            enqueued += 1

        return enqueued

    # ----------------------------- finalize ----------------------------- #

    async def _finalize(
        self,
        conn: aiosqlite.Connection,
        deps: AgentDeps,
        session: Session,
        stop_reason: StopReason | None,
    ) -> None:
        n_cancel = await task_repo.cancel_pending_for_session(conn, session.id)
        if n_cancel:
            log.info("pending_cancelled", n=n_cancel)

        # Machine-readable Elo standings — the handoff reads this to pick the
        # tournament winner instead of globbing files in mtime order.
        try:
            await self._write_standings(conn, session)
        except Exception as e:
            log.exception("standings_write_failed", err=str(e))

        # Try to run the proper final overview via metareview if the agent exists.
        # Fall back to the stub if metareview is not yet wired in (older builds).
        try:
            from .metareview import MetaReviewAgent

            agent = MetaReviewAgent(deps)
            final_task = Task(
                id=ids.task_id(), session_id=session.id,
                created_at=datetime.now(UTC),
                agent="metareview", action="GenerateFinalResearchOverview",
                target_id=None, payload={}, priority=1, status="pending",
                idempotency_key=f"{session.id}::metareview::final",
            )
            await task_repo.enqueue(conn, final_task)
            await task_repo.mark_in_progress(conn, final_task.id)
            try:
                result = await agent.execute(final_task)
                overview_path = result.extra.get("overview_path")
                if overview_path:
                    await sess_repo.set_final_overview(conn, session.id, overview_path)
                await task_repo.complete(conn, final_task.id)
            except Exception as e:
                log.exception("final_overview_failed", err=str(e))
                await task_repo.fail(conn, final_task.id, error=str(e),
                                      max_attempts=self.cfg.lease.max_attempts)
                overview_path = await self._write_simple_overview(conn, session)
                await sess_repo.set_final_overview(conn, session.id, overview_path)
        except ImportError:
            overview_path = await self._write_simple_overview(conn, session)
            await sess_repo.set_final_overview(conn, session.id, overview_path)

        # `set_final_overview` flips status to 'done' atomically. If the
        # overview path was never set (e.g. metareview crashed and the simple
        # overview also failed) the status is still 'running'; force-set it
        # here so the session doesn't appear to be running forever after exit.
        # For EXTERNAL stops we don't overwrite the user-set 'paused' /
        # 'aborted' status.
        if stop_reason != StopReason.EXTERNAL:
            await sess_repo.set_status(conn, session.id, "done")

        await self._emit(conn, session.id, "session_done",
                         {"stop_reason": stop_reason.value if stop_reason else None})

    async def _write_standings(
        self, conn: aiosqlite.Connection, session: Session
    ) -> str:
        """Write final/standings.json — the full Elo table, ranked."""
        from ..storage.artifacts import write_json
        from ..storage.repos import tournaments as tourney_repo

        hyps = await hyp_repo.list_for_session(conn, session.id)
        hyps = sorted(hyps, key=lambda h: -(h.elo if h.elo is not None else -1))
        standings = {
            "session_id": session.id,
            "match_count": await tourney_repo.count_matches(conn, session.id),
            "standings": [
                {
                    "rank": i,
                    "id": h.id,
                    "elo": h.elo,
                    "matches_played": h.matches_played,
                    "state": h.state,
                    "strategy": h.strategy,
                    "created_by": h.created_by,
                    "title": h.title,
                    "artifact_path": h.artifact_path,
                }
                for i, h in enumerate(hyps, 1)
            ],
        }
        path = await write_json(self.cfg, session.id, "final", "standings", standings)
        log.info("standings_written", path=path, n=len(hyps))
        return path

    async def _write_simple_overview(
        self, conn: aiosqlite.Connection, session: Session
    ) -> str:
        hyps = await hyp_repo.list_for_session(conn, session.id)
        parts: list[str] = [
            f"# Research overview — session {session.id}",
            f"\n**Goal.** {session.research_goal}\n",
            f"**Hypotheses produced.** {len(hyps)}",
            "",
        ]
        for i, h in enumerate(hyps, 1):
            parts.append(f"## {i}. {h.title or h.id}")
            parts.append(
                f"`{h.id}` — strategy `{h.strategy}` — state `{h.state}` "
                f"— Elo `{h.elo:.0f}`" if h.elo is not None else
                f"`{h.id}` — strategy `{h.strategy}` — state `{h.state}`"
            )
            parts.append(h.summary or "(no summary)")
            reviews = await rev_repo.list_for_hypothesis(conn, h.id)
            if reviews:
                parts.append("\n**Reviews:**")
                for r in reviews:
                    parts.append(
                        f"- *{r.kind}* — verdict `{r.verdict or '?'}` "
                        f"(n={r.scores.novelty}, c={r.scores.correctness}, "
                        f"t={r.scores.testability})"
                    )
            parts.append("")
        body = "\n".join(parts)
        return await write_text(self.cfg, session.id, "final", "overview", ".md", body)

    # ----------------------------- helpers ----------------------------- #

    def _build_agents(self, deps: AgentDeps) -> dict[str, object]:
        out: dict[str, object] = {
            "generation": GenerationAgent(deps),
            "reflection": ReflectionAgent(deps),
            "ranking": RankingAgent(deps),
        }
        # Evolution / Proximity / Meta-review register if importable.
        try:
            from .evolution import EvolutionAgent

            out["evolution"] = EvolutionAgent(deps)
        except ImportError:
            pass
        try:
            from .proximity import ProximityAgent

            out["proximity"] = ProximityAgent(deps)
        except ImportError:
            pass
        try:
            from .metareview import MetaReviewAgent

            out["metareview"] = MetaReviewAgent(deps)
        except ImportError:
            pass
        return out

    async def _emit(
        self,
        conn: aiosqlite.Connection,
        session_id: str,
        event: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        await events_repo.emit(
            conn, session_id=session_id, task_id=None, agent="supervisor",
            event=event, payload=payload,
        )
        await GLOBAL_BUS.publish(session_id, event, payload)


# ----------------------------- helpers ----------------------------- #


def _human_preference(session_id: str, text: str):
    from ..models import SystemFeedback

    return SystemFeedback(
        id=ids.feedback_id(), session_id=session_id,
        created_at=datetime.now(UTC),
        source="human", kind="preference",
        target_id=None, text=text, active=True,
    )
