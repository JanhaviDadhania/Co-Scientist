"""Tournament matches + Elo journal.

The Elo update is the *only* place hypotheses.elo and matches_played mutate
during a session. It runs in a single transaction guarded by elo_journal.match_id
UNIQUE — so re-running the same match is a no-op.
"""

from __future__ import annotations

import time
from datetime import datetime

import aiosqlite

from ...models import TournamentMatch


async def insert_match(conn: aiosqlite.Connection, m: TournamentMatch) -> bool:
    """Insert the descriptive match row. Idempotent by id."""
    cur = await conn.execute(
        """INSERT OR IGNORE INTO tournament_matches(
               id, session_id, created_at, hyp_a, hyp_b, mode, winner,
               elo_a_before, elo_b_before, elo_a_after, elo_b_after,
               rationale, transcript_id, similarity)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            m.id, m.session_id, m.created_at.isoformat(),
            m.hyp_a, m.hyp_b, m.mode, m.winner,
            m.elo_a_before, m.elo_b_before, m.elo_a_after, m.elo_b_after,
            m.rationale, m.transcript_id, m.similarity,
        ),
    )
    ok = cur.rowcount > 0
    await conn.commit()
    return ok


async def apply_elo_update(
    conn: aiosqlite.Connection,
    *,
    match_id: str,
    hyp_a: str,
    hyp_b: str,
    winner: str,
    elo_a_before: float,
    elo_b_before: float,
    elo_a_after: float,
    elo_b_after: float,
) -> bool:
    """Apply an Elo update atomically and idempotently.

    Returns True if the update was newly applied; False if the journal already
    has this match_id (re-run; we skip).
    """
    applied_at = int(time.time() * 1000)
    try:
        await conn.execute("BEGIN IMMEDIATE")
        await conn.execute(
            """INSERT INTO elo_journal(
                   update_id, match_id, hyp_a, hyp_b, winner,
                   elo_a_before, elo_b_before, elo_a_after, elo_b_after, applied_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (match_id, match_id, hyp_a, hyp_b, winner,
             elo_a_before, elo_b_before, elo_a_after, elo_b_after, applied_at),
        )
        await conn.execute(
            """UPDATE hypotheses
                  SET elo=?, matches_played=matches_played+1
                WHERE id=?""",
            (elo_a_after, hyp_a),
        )
        await conn.execute(
            """UPDATE hypotheses
                  SET elo=?, matches_played=matches_played+1
                WHERE id=?""",
            (elo_b_after, hyp_b),
        )
        # also reflect the after-Elo back into tournament_matches row if it exists
        await conn.execute(
            """UPDATE tournament_matches
                  SET elo_a_after=?, elo_b_after=?
                WHERE id=?""",
            (elo_a_after, elo_b_after, match_id),
        )
        await conn.commit()
        return True
    except aiosqlite.IntegrityError:
        # match_id already in journal — idempotent skip.
        await conn.rollback()
        return False


async def count_matches(conn: aiosqlite.Connection, session_id: str) -> int:
    async with conn.execute(
        "SELECT COUNT(*) AS n FROM tournament_matches WHERE session_id=? AND mode != 'invalid'",
        (session_id,),
    ) as cur:
        row = await cur.fetchone()
    return row["n"] if row else 0


async def winning_rationale_for(
    conn: aiosqlite.Connection, session_id: str, hypothesis_id: str
) -> str | None:
    """The most recent debate rationale from a match this hypothesis WON."""
    async with conn.execute(
        """SELECT rationale FROM tournament_matches
              WHERE session_id=? AND rationale IS NOT NULL
                AND ((winner='a' AND hyp_a=?) OR (winner='b' AND hyp_b=?))
              ORDER BY created_at DESC LIMIT 1""",
        (session_id, hypothesis_id, hypothesis_id),
    ) as cur:
        row = await cur.fetchone()
    return row["rationale"] if row else None


async def recent_rationales(
    conn: aiosqlite.Connection, session_id: str, limit: int = 50
) -> list[str]:
    async with conn.execute(
        """SELECT rationale FROM tournament_matches
              WHERE session_id=? AND rationale IS NOT NULL
              ORDER BY created_at DESC LIMIT ?""",
        (session_id, limit),
    ) as cur:
        rows = await cur.fetchall()
    return [r["rationale"] for r in rows]


def _row_to_match(row: aiosqlite.Row) -> TournamentMatch:
    return TournamentMatch(
        id=row["id"],
        session_id=row["session_id"],
        created_at=datetime.fromisoformat(row["created_at"]),
        hyp_a=row["hyp_a"], hyp_b=row["hyp_b"], mode=row["mode"],
        winner=row["winner"],
        elo_a_before=row["elo_a_before"], elo_b_before=row["elo_b_before"],
        elo_a_after=row["elo_a_after"], elo_b_after=row["elo_b_after"],
        rationale=row["rationale"], transcript_id=row["transcript_id"],
        similarity=row["similarity"],
    )
