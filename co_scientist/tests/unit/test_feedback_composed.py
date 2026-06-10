"""Tests for the composed feedback pull (human rows + latest meta-review)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from co_scientist import ids
from co_scientist.models import ResearchPlan, Session, SystemFeedback
from co_scientist.storage.repos import feedback as fb_repo
from co_scientist.storage.repos import sessions as sess_repo

pytestmark = pytest.mark.asyncio


async def _mk_session(conn) -> str:
    sid = ids.session_id()
    now = datetime.now(UTC)
    await sess_repo.insert(conn, Session(
        id=sid, created_at=now, updated_at=now, status="running",
        research_goal="g", research_plan=ResearchPlan(objective="g"),
        config_snapshot={}, budget_tokens=1, budget_usd=1.0,
    ))
    return sid


def _fb(sid: str, *, source: str, kind: str, text: str,
        target: str | None = None, active: bool = True,
        at: datetime | None = None) -> SystemFeedback:
    return SystemFeedback(
        id=ids.feedback_id(), session_id=sid,
        created_at=at or datetime.now(UTC),
        source=source, kind=kind, target_id=target, text=text, active=active,
    )


async def test_composed_is_none_when_empty(conn) -> None:
    sid = await _mk_session(conn)
    assert await fb_repo.composed_feedback(conn, sid) is None


async def test_composed_includes_all_active_human_rows(conn) -> None:
    sid = await _mk_session(conn)
    await fb_repo.insert(conn, _fb(sid, source="human", kind="directive",
                                   text="focus on CPU-feasible mechanisms"))
    await fb_repo.insert(conn, _fb(sid, source="human", kind="rejection",
                                   text="drop this idea", target="hyp_X"))
    out = await fb_repo.composed_feedback(conn, sid)
    assert out is not None
    assert "focus on CPU-feasible mechanisms" in out
    assert "[rejection] (re: hyp_X)" in out


async def test_composed_human_rows_survive_newer_metareview(conn) -> None:
    sid = await _mk_session(conn)
    t0 = datetime.now(UTC)
    await fb_repo.insert(conn, _fb(sid, source="human", kind="directive",
                                   text="HUMAN-DIRECTIVE", at=t0))
    await fb_repo.insert(conn, _fb(sid, source="meta_review", kind="system_feedback",
                                   text="META-OLD", at=t0 + timedelta(seconds=1)))
    await fb_repo.insert(conn, _fb(sid, source="meta_review", kind="system_feedback",
                                   text="META-NEW", at=t0 + timedelta(seconds=2)))
    out = await fb_repo.composed_feedback(conn, sid)
    assert out is not None
    assert "HUMAN-DIRECTIVE" in out      # never shadowed
    assert "META-NEW" in out             # latest steering
    assert "META-OLD" not in out         # only the latest meta row


async def test_inactive_rows_are_excluded(conn) -> None:
    sid = await _mk_session(conn)
    fb = _fb(sid, source="human", kind="directive", text="RETIRED")
    await fb_repo.insert(conn, fb)
    await fb_repo.deactivate(conn, fb.id)
    assert await fb_repo.composed_feedback(conn, sid) is None


async def test_latest_system_feedback_honors_active_flag(conn) -> None:
    sid = await _mk_session(conn)
    fb = _fb(sid, source="meta_review", kind="system_feedback", text="STEER")
    await fb_repo.insert(conn, fb)
    assert (await fb_repo.latest_system_feedback(conn, sid)).text == "STEER"
    await fb_repo.deactivate(conn, fb.id)
    assert await fb_repo.latest_system_feedback(conn, sid) is None
