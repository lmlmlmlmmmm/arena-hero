"""Deterministic pathfinding and movement helpers for the Arena Hero tactic."""

from __future__ import annotations

from heapq import heappop, heappush
from typing import Protocol

from arena_hero import Direction, Position

from tactic_config import (
    DIRECTION_DELTAS,
    DIRECTION_ORDER,
    PATHFIND_BUDGET,
    ROUTE_ESTIMATE_BUDGET,
)


class ReservationLedger(Protocol):
    destinations: set[Position]

    def reserve(self, unit_id, destination: Position) -> None: ...


def manhattan(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def direction_between(a: Position, b: Position) -> Direction:
    if b[1] < a[1]:
        return Direction.UP
    if b[1] > a[1]:
        return Direction.DOWN
    if b[0] < a[0]:
        return Direction.LEFT
    return Direction.RIGHT


def _monotonic_route_clear(
    start: Position,
    goal: Position,
    blocked: set[Position] | frozenset[Position],
    reserved: set[Position] | frozenset[Position],
    first_axis: int,
) -> bool:
    """Check one x-then-y or y-then-x route without allocating a path."""

    x, y = start
    axes = (0, 1) if first_axis == 0 else (1, 0)
    for axis in axes:
        target = goal[axis]
        while (x, y)[axis] != target:
            dx = 1 if goal[0] > x else -1 if goal[0] < x else 0
            dy = 1 if goal[1] > y else -1 if goal[1] < y else 0
            if axis == 0:
                x += dx
            else:
                y += dy
            cell = (x, y)
            if cell in reserved:
                return False
            if cell != goal and cell in blocked:
                return False
    return True


def bounded_route_cost(
    start: Position,
    goal: Position,
    blocked: set[Position] | frozenset[Position],
) -> int | None:
    """Estimate a route length, returning None for a costly or unknown route."""

    direct = manhattan(start, goal)
    if direct == 0:
        return 0
    if _monotonic_route_clear(start, goal, blocked, frozenset(), 0):
        return direct
    if _monotonic_route_clear(start, goal, blocked, frozenset(), 1):
        return direct

    frontier: list[tuple[int, int, Position]] = [(direct, 0, start)]
    best_cost = {start: 0}
    budget = ROUTE_ESTIMATE_BUDGET
    while frontier and budget > 0:
        budget -= 1
        _, cost, current = heappop(frontier)
        if cost != best_cost.get(current):
            continue
        if current == goal:
            return cost
        for direction in DIRECTION_ORDER:
            dx, dy = DIRECTION_DELTAS[direction]
            nxt = (current[0] + dx, current[1] + dy)
            if nxt != goal and nxt in blocked:
                continue
            next_cost = cost + 1
            estimate = next_cost + manhattan(nxt, goal)
            if next_cost >= best_cost.get(nxt, float("inf")):
                continue
            best_cost[nxt] = next_cost
            heappush(frontier, (estimate, next_cost, nxt))
    if frontier:
        return min(estimate for estimate, _, _ in frontier)
    return None


def step_toward(
    start: Position,
    goal: Position,
    blocked: set[Position],
    reserved: set[Position] | frozenset[Position] = frozenset(),
) -> Position | None:
    """Return the first deterministic pathfinding step toward a goal."""

    if start == goal or goal in reserved:
        return None

    if _monotonic_route_clear(start, goal, blocked, reserved, 0):
        dx = 1 if goal[0] > start[0] else -1 if goal[0] < start[0] else 0
        if dx:
            return (start[0] + dx, start[1])
        dy = 1 if goal[1] > start[1] else -1
        return (start[0], start[1] + dy)
    if _monotonic_route_clear(start, goal, blocked, reserved, 1):
        dy = 1 if goal[1] > start[1] else -1 if goal[1] < start[1] else 0
        if dy:
            return (start[0], start[1] + dy)
        dx = 1 if goal[0] > start[0] else -1
        return (start[0] + dx, start[1])

    frontier: list[tuple[int, int, int, int, Position, Position | None]] = []
    distance = manhattan(start, goal)
    heappush(frontier, (distance, distance, 0, 0, start, None))
    best_cost = {start: 0}
    discovery_order = 0
    budget = PATHFIND_BUDGET
    while frontier and budget > 0:
        budget -= 1
        _, _, cost, _, current, first_step = heappop(frontier)
        if cost != best_cost.get(current):
            continue
        if current == goal:
            return first_step
        for direction in DIRECTION_ORDER:
            dx, dy = DIRECTION_DELTAS[direction]
            nxt = (current[0] + dx, current[1] + dy)
            if nxt in reserved:
                continue
            if nxt != goal and nxt in blocked:
                continue
            next_cost = cost + 1
            if next_cost >= best_cost.get(nxt, float("inf")):
                continue
            best_cost[nxt] = next_cost
            discovery_order += 1
            heappush(
                frontier,
                (
                    next_cost + manhattan(nxt, goal),
                    manhattan(nxt, goal),
                    next_cost,
                    discovery_order,
                    nxt,
                    first_step if first_step is not None else nxt,
                ),
            )
    if frontier:
        *_, first_step = min(frontier, key=lambda item: (item[1], item[2], item[3]))
        return first_step
    return None


def move_or_wait(
    unit,
    goal: Position,
    blocked: set[Position],
    reservations: ReservationLedger | None = None,
) -> bool:
    """Queue a non-conflicting step toward a goal, or wait."""

    reserved = reservations.destinations if reservations is not None else frozenset()
    step = step_toward(unit.position, goal, blocked, reserved)
    if step is None:
        unit.wait()
        return False
    unit.move(direction_between(unit.position, step))
    if reservations is not None:
        reservations.reserve(unit.id, step)
    return True


def move_or_escape(
    unit,
    goal: Position,
    blocked: set[Position],
    hard_blocked: set[Position],
    reservations: ReservationLedger | None = None,
) -> bool:
    """Prefer a safe route, but never freeze cargo inside a visible threat."""

    reserved = reservations.destinations if reservations is not None else frozenset()
    step = step_toward(unit.position, goal, blocked, reserved)
    if step is None and unit.position in blocked and unit.position not in hard_blocked:
        candidates: list[tuple[int, int, int, Position]] = []
        for order, direction in enumerate(DIRECTION_ORDER):
            dx, dy = DIRECTION_DELTAS[direction]
            candidate = (unit.position[0] + dx, unit.position[1] + dy)
            if candidate in hard_blocked or candidate in reserved:
                continue
            candidates.append(
                (
                    1 if candidate in blocked else 0,
                    manhattan(candidate, goal),
                    order,
                    candidate,
                )
            )
        if candidates:
            *_, step = min(candidates)
    if step is None:
        unit.wait()
        return False
    unit.move(direction_between(unit.position, step))
    if reservations is not None:
        reservations.reserve(unit.id, step)
    return True


def nearest_target(origin: Position, targets: set[Position]) -> Position | None:
    ordered = sorted(targets, key=lambda cell: (manhattan(origin, cell), cell[0], cell[1]))
    return ordered[0] if ordered else None


__all__ = [
    "bounded_route_cost",
    "direction_between",
    "manhattan",
    "move_or_escape",
    "move_or_wait",
    "nearest_target",
    "step_toward",
]
