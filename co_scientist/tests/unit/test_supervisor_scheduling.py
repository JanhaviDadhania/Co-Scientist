"""Unit tests for batched generation sizing and proportional gates."""

from __future__ import annotations

from co_scientist.agents.supervisor import (
    DEFAULT_N_IDEAS,
    evolution_gate,
    metareview_due,
    next_batch_size,
)


def test_default_n_ideas_is_15() -> None:
    assert DEFAULT_N_IDEAS == 15


def test_batch_schedule_for_15_is_8_4_3() -> None:
    sizes = []
    remaining = 15
    while remaining > 0:
        b = next_batch_size(remaining)
        sizes.append(b)
        remaining -= b
    assert sizes == [8, 4, 3]


def test_batch_small_n_fires_all_at_once() -> None:
    assert next_batch_size(3) == 3
    assert next_batch_size(4) == 4
    assert next_batch_size(1) == 1


def test_batch_zero_or_negative_remaining() -> None:
    assert next_batch_size(0) == 0
    assert next_batch_size(-2) == 0


def test_evolution_gate_proportional_and_capped() -> None:
    assert evolution_gate(3) == 4      # floor: small pops still need 4
    assert evolution_gate(8) == 4
    assert evolution_gate(15) == 8
    assert evolution_gate(30) == 10    # capped at 10
    assert evolution_gate(100) == 10   # never more than 10


def test_evolution_gate_reachable_from_population_8() -> None:
    # The old flat-20 gate was unreachable at realistic sizes; the new gate
    # must be satisfiable by the population itself.
    for pop in (8, 15, 25, 60):
        assert evolution_gate(pop) <= pop


def test_metareview_due_proportional() -> None:
    # population 8 → interval 32
    assert not metareview_due(31, 0, 8)
    assert metareview_due(32, 0, 8)
    # second feedback needs 2× the interval
    assert not metareview_due(63, 1, 8)
    assert metareview_due(64, 1, 8)


def test_metareview_not_due_for_tiny_population() -> None:
    assert not metareview_due(100, 0, 1)
    assert not metareview_due(100, 0, 0)
