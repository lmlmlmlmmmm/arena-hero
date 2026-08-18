"""Balanced Arena Hero tactic: economy, defense, Beacon control, and upkeep.

Reads the API key from the ARENA_HERO_API_KEY environment variable or from the
file given by --env, connects with the official SDK, and submits one plan per
Tick. Terrain knowledge and scouting progress survive restarts through the
--state file.
"""

from __future__ import annotations

import argparse
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from heapq import heappop, heappush
from pathlib import Path
from uuid import UUID

from arena_hero import (
    APIError,
    ArenaHeroClient,
    BeaconStatus,
    ChampionBeacon,
    CoreState,
    CoreView,
    Direction,
    MoveAction,
    PlayerStatus,
    Position,
    UnitType,
    UnitView,
    core_resource_capacity,
    unit_cost,
)

log = logging.getLogger("arena-hero-tactic")

DIRECTION_ORDER = (Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT)
# ``Direction.delta`` is an SDK property.  Cache it once instead of rebuilding
# the same tiny mapping inside every A* expansion and planning loop.
DIRECTION_DELTAS = {direction: direction.delta for direction in DIRECTION_ORDER}
MIN_ECONOMY_WORKERS = 4
MAX_WORKER_POPULATION = 14
WORKERS_PER_GUARD = 3
BASE_PRICE_POPULATION_LIMIT = 20
CORE_HEAL_RESERVE = 2
CORE_HP_FLOOR = 5
CORE_SHIELD_FLOOR = 5
CRITICAL_CORE_HP = 2
DEFENDER_SPAWN_DISTANCE = 4
DANGER_DISTANCE = 3
PATHFIND_BUDGET = 8192
ROUTE_ESTIMATE_BUDGET = 2048
PATH_COST_UNREACHABLE = 1_000_000
# A normal trip must remain close enough that one unit of cargo can return in a
# useful time.  Leg limits protect pathfinding; the adaptive round-trip budget
# below is the economic limit used by the live policy.
RESOURCE_MAX_ASSIGNMENT_COST = 64
RESOURCE_LOCAL_RETURN_DISTANCE = 24
RESOURCE_REMOTE_WORKERS_PER_FLEET = 4
RESOURCE_ADAPTIVE_REMOTE_MIN_WORKERS = 6
RESOURCE_ADAPTIVE_REMOTE_LIMIT = 2
RESOURCE_TRIP_COST_HEALTHY = 72
RESOURCE_TRIP_COST_NORMAL = 60
RESOURCE_TRIP_COST_RECOVERY = 48
ECONOMY_FLOW_WINDOW = 64
ECONOMY_HISTORY_LIMIT = 128
# Enough recorded Ticks to call a drought a drought rather than a quiet stretch.
STARVATION_SAMPLES = 16

# Absolute grid coverage is retained in memory, while actual waypoints stay in
# a bounded ring around the current Core.
SCOUT_OFFSETS = (
    (0, -1),
    (1, -1),
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
)
SCOUT_MIN_RADIUS = 4
SCOUT_RADIUS_STEP = 5
SCOUT_RING_COUNT = 8
SCOUT_ARRIVAL_DISTANCE = 1
SCOUT_GRID_SIZE = 3
SCOUT_MAX_DISTANCE = 40
SCOUT_TETHER_DISTANCE = 48
# A wider disc only helps once the near one is genuinely used up: the same lone
# scout spread over four times the area covers it four times more slowly.  Grow
# the search only after most of the near grid has actually been seen, and keep
# the recall tether the same distance beyond whichever radius is in force.
SCOUT_EXHAUSTED_RATIO = 0.8
SCOUT_FAR_DISTANCE = 72
SCOUT_TETHER_MARGIN = SCOUT_TETHER_DISTANCE - SCOUT_MAX_DISTANCE
SCOUT_SAFE_RETURN_DISTANCE = 12
SCOUT_COVERAGE_TTL = 4096
SCOUT_COVERAGE_MAX_CELLS = 32768
SCOUT_HISTORY_LIMIT = 8
SCOUT_LOOP_WINDOW = 6
# A Worker standing perfectly still appends nothing to its position history, so
# oscillation detection alone cannot see it.  Count the idle Ticks separately.
WORKER_STALL_TICKS = 6
SCOUT_ABSOLUTE_GRID_SCHEMA = 3
SCOUT_STATE_SCHEMA = 5
SCOUT_SAVE_INTERVAL = 8
# Obstacle terrain is permanent, so it is never expired by age.  It is still
# bounded, because a long game walks the Core far enough to accumulate terrain
# that no longer influences any route.  Pruning drops the cells farthest from
# the Core, and only down to KEEP, so the sort runs rarely instead of per Tick.
OBSTACLE_MEMORY_MAX_CELLS = 32768
OBSTACLE_MEMORY_KEEP_CELLS = 28672
TACTIC_LOG_MAX_BYTES = 5 * 1024 * 1024
TACTIC_LOG_BACKUPS = 3

DEPLETED_TTL = 40
RESOURCE_MEMORY_TTL = 64
RESOURCE_ACTIVE_ASSIGNMENT_TTL = 128
RESOURCE_REASSIGN_BONUS = 2
RESOURCE_STALE_PENALTY_MAX = 6
RESOURCE_STALL_TICKS = 6
RESOURCE_COOLDOWN_TICKS = 8
CONTESTED_CELL_TTL = 4
# Enemy vision flickers, so a threatened cell stays on the avoid list for a few
# Ticks after the shooter drops out of sight.  Without that hysteresis a loaded
# Worker oscillates between the same two cells forever.
THREAT_MEMORY_TICKS = 6
CORE_VISION_RADIUS = 5
UNIT_VISION_RADII = {
    UnitType.WORKER: 3,
    UnitType.VANGUARD: 4,
    UnitType.RANGER: 5,
}

BEACON_CLAIM_DISTANCE = 12
BEACON_HOLD_DISTANCE = 1
# Diverting our only Unit to the Beacon would stall the economy outright.
BEACON_MIN_UNITS = 2

CORE_MIGRATION_TICKS = 4
CORE_MIGRATION_INCOMING_MARGIN = 2
CORE_RELOCATION_MIN_DISTANCE = 8
CORE_RELOCATION_MIN_WORKERS = 4
CORE_RELOCATION_MIN_ACTIVE_WORKERS = 2
CORE_RELOCATION_ENEMY_BUFFER = 8
CORE_MIGRATION_PRODUCTION_GAP = 2
CORE_DEFENSE_ALERT_DISTANCE = 8
CORE_SHIELD_EMERGENCY_FLOOR = 2
CORE_THREAT_CAUTION_TICKS = 6
CORE_PREFERRED_RESOURCE_QUOTA = 6
# A Worker has two hit points and sees three cells, so it can neither win nor
# outrun a fight it has already walked into.  Leave before contact, not after.
WORKER_FLEE_DISTANCE = 4

GUARD_OFFSETS = (
    (0, -2),
    (2, 0),
    (0, 2),
    (-2, 0),
    (2, -2),
    (2, 2),
    (-2, 2),
    (-2, -2),
)
PATROL_OFFSETS = SCOUT_OFFSETS
PATROL_ROTATION_TICKS = 8
RANGER_PATROL_RADIUS = 8
VANGUARD_PATROL_RADIUS = 5

# Event types carrying one of these markers mean the server rejected or failed
# something we asked for, which is always worth surfacing in the log.
FAILURE_MARKERS = ("FAILED", "REJECTED", "INVALID", "DENIED", "MISSED", "OVERFLOW")
STALE_RESOURCE_REASONS = {"RESOURCE_DEPLETED", "NOT_RESOURCE_CELL", "NODE_EXHAUSTED"}


@dataclass
class ResourceProgress:
    """Best known route cost and consecutive Ticks without improvement."""

    target: Position
    best_cost: int
    stalled_ticks: int = 0


@dataclass
class ScoutMemory:
    """Cross-Tick state: permanent terrain, resources, scouting, and events."""

    offsets: dict[str, int] = field(default_factory=dict)
    sweeps: dict[str, int] = field(default_factory=dict)
    known_obstacles: set[Position] = field(default_factory=set)
    known_resources: set[Position] = field(default_factory=set)
    resource_last_seen: dict[Position, int] = field(default_factory=dict)
    scout_seen: dict[tuple[int, int], int] = field(default_factory=dict)
    scout_targets: dict[str, tuple[int, int]] = field(default_factory=dict)
    scout_positions: dict[str, list[Position]] = field(default_factory=dict)
    position_stalls: dict[str, int] = field(default_factory=dict)
    resource_assignments: dict[str, Position] = field(default_factory=dict)
    resource_progress: dict[str, ResourceProgress] = field(default_factory=dict)
    resource_cooldowns: dict[tuple[str, Position], int] = field(default_factory=dict)
    recalling_workers: set[str] = field(default_factory=set)
    last_move_destinations: dict[str, Position] = field(default_factory=dict)
    depleted: dict[Position, int] = field(default_factory=dict)
    contested: dict[Position, int] = field(default_factory=dict)
    threatened: dict[Position, int] = field(default_factory=dict)
    last_events: Counter = field(default_factory=Counter)
    last_intents: Counter = field(default_factory=Counter)
    last_resource_flow: Counter = field(default_factory=Counter)
    economic_history: list[tuple[int, int, int, int]] = field(default_factory=list)
    last_trip_budget: int = RESOURCE_TRIP_COST_NORMAL
    core_threat_until_tick: int = 0
    core_intercept_worker_id: str | None = None
    last_migration_hold: str | None = None
    last_defense_status: str | None = None
    path: Path | None = None
    dirty: bool = False
    last_saved_tick: int = -1

    def load(self) -> None:
        """Restore state field-by-field, preserving good data after corruption.

        ``scout_seen`` and per-worker waypoints were relative to the Core before
        schema 2, and schema 2 used a coarser grid.  Neither representation can
        safely be reused by schema 3, so only scouting coverage is discarded;
        absolute terrain, resource memory, and newer optional metrics survive.
        """

        if self.path is None or not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text())
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            log.warning("unusable scout state at %s; starting fresh", self.path)
            return

        if not isinstance(raw, dict):
            log.warning("unusable scout state at %s; starting fresh", self.path)
            return

        def parse_position(value) -> Position | None:
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                return None
            try:
                return (int(value[0]), int(value[1]))
            except (TypeError, ValueError, OverflowError):
                return None

        def parse_position_set(value) -> set[Position]:
            result: set[Position] = set()
            if not isinstance(value, list):
                return result
            for item in value:
                position = parse_position(item)
                if position is not None:
                    result.add(position)
            return result

        def parse_position_map(value) -> dict[str, Position]:
            result: dict[str, Position] = {}
            if not isinstance(value, dict):
                return result
            for key, item in value.items():
                position = parse_position(item)
                if position is not None:
                    result[str(key)] = position
            return result

        def parse_int_map(value) -> dict[str, int]:
            result: dict[str, int] = {}
            if not isinstance(value, dict):
                return result
            for key, item in value.items():
                # bool is technically an int in Python, but never a useful
                # tick/offset value in this file.
                if isinstance(item, int) and not isinstance(item, bool):
                    result[str(key)] = item
            return result

        def parse_cell_int_map(value) -> dict[Position, int]:
            """Decode the ``"x,y" -> tick`` form used by every cell-keyed map."""

            result: dict[Position, int] = {}
            if not isinstance(value, dict):
                return result
            for encoded, number in value.items():
                if not isinstance(encoded, str) or "," not in encoded:
                    continue
                parts = encoded.split(",", 1)
                try:
                    cell = (int(parts[0]), int(parts[1]))
                except (TypeError, ValueError, OverflowError):
                    continue
                if isinstance(number, int) and not isinstance(number, bool):
                    result[cell] = number
            return result

        def parse_resource_progress(value) -> dict[str, ResourceProgress]:
            result: dict[str, ResourceProgress] = {}
            if not isinstance(value, dict):
                return result
            for worker_id, item in value.items():
                if not isinstance(item, (list, tuple)) or len(item) != 4:
                    continue
                if not all(
                    isinstance(number, int) and not isinstance(number, bool)
                    for number in item
                ):
                    continue
                result[str(worker_id)] = ResourceProgress(
                    (item[0], item[1]), item[2], max(0, item[3])
                )
            return result

        schema = raw.get("schema_version", 1)
        if not isinstance(schema, int):
            schema = 1

        # Absolute terrain/resource coordinates are safe to retain even from
        # the pre-schema format.  Parse each field independently so one bad
        # entry cannot throw away hundreds of valid obstacle cells.
        self.known_obstacles = parse_position_set(raw.get("known_obstacles", []))
        self.known_resources = parse_position_set(raw.get("known_resources", []))
        self.resource_last_seen = parse_cell_int_map(raw.get("resource_last_seen", {}))
        # Ticks are server-global and keep counting across a restart, so these
        # absolute deadlines stay meaningful; anything already past expires on
        # the first expire() call rather than misleading the planner.
        self.depleted = parse_cell_int_map(raw.get("depleted", {}))
        self.contested = parse_cell_int_map(raw.get("contested", {}))
        self.resource_progress = parse_resource_progress(raw.get("resource_progress", {}))
        self.resource_assignments = parse_position_map(raw.get("resource_assignments", {}))
        self.last_move_destinations = parse_position_map(
            raw.get("last_move_destinations", {})
        )
        raw_core_threat_until = raw.get("core_threat_until_tick", 0)
        self.core_threat_until_tick = (
            raw_core_threat_until
            if isinstance(raw_core_threat_until, int)
            and not isinstance(raw_core_threat_until, bool)
            else 0
        )
        raw_intercept_worker_id = raw.get("core_intercept_worker_id")
        self.core_intercept_worker_id = (
            raw_intercept_worker_id
            if isinstance(raw_intercept_worker_id, str) and raw_intercept_worker_id
            else None
        )
        self.economic_history = []
        raw_economic_history = raw.get("economic_history", [])
        if isinstance(raw_economic_history, list):
            samples: dict[int, tuple[int, int, int, int]] = {}
            for item in raw_economic_history:
                if not isinstance(item, (list, tuple)) or len(item) != 4:
                    continue
                if not all(isinstance(value, int) and not isinstance(value, bool) for value in item):
                    continue
                sample = tuple(max(0, value) for value in item)
                samples[sample[0]] = sample
            self.economic_history = [samples[tick] for tick in sorted(samples)][
                -ECONOMY_HISTORY_LIMIT:
            ]

        if schema < SCOUT_ABSOLUTE_GRID_SCHEMA:
            # Relative keys and waypoints cannot be migrated without the Core
            # position at the time they were written.  Start a fresh absolute
            # frontier, but keep the useful permanent maps above.
            self.offsets = {}
            self.sweeps = {}
            self.scout_seen = {}
            self.scout_targets = {}
            self.scout_positions = {}
            self.position_stalls = {}
            self.dirty = True
            log.info(
                "migrating legacy scout state at %s; retained obstacles=%d resources=%d",
                self.path,
                len(self.known_obstacles),
                len(self.known_resources),
            )
        else:
            self.offsets = parse_int_map(raw.get("offsets", {}))
            self.sweeps = parse_int_map(raw.get("sweeps", {}))
            self.scout_seen = parse_cell_int_map(raw.get("scout_seen", {}))
            self.scout_targets = parse_position_map(raw.get("scout_targets", {}))
            self.position_stalls = {
                worker_id: max(0, count)
                for worker_id, count in parse_int_map(
                    raw.get("position_stalls", {})
                ).items()
            }
            self.scout_positions = {}
            raw_positions = raw.get("scout_positions", {})
            if isinstance(raw_positions, dict):
                for worker_id, positions in raw_positions.items():
                    if not isinstance(positions, list):
                        continue
                    parsed = [
                        position
                        for item in positions
                        if (position := parse_position(item)) is not None
                    ]
                    if parsed:
                        self.scout_positions[str(worker_id)] = parsed[-SCOUT_HISTORY_LIMIT:]

        log.info(
            "restored scout progress for %d workers with %d known obstacles",
            len(self.offsets),
            len(self.known_obstacles),
        )

    def save(self, tick: int | None = None, *, force: bool = False) -> None:
        """Atomically persist changed state, throttled during live play."""

        if self.path is None or not self.dirty:
            return
        if (
            not force
            and tick is not None
            and self.last_saved_tick >= 0
            and tick - self.last_saved_tick < SCOUT_SAVE_INTERVAL
        ):
            return
        temporary = self.path.with_name(self.path.name + ".tmp")
        try:
            temporary.write_text(
                json.dumps(
                    {
                        "schema_version": SCOUT_STATE_SCHEMA,
                        "offsets": self.offsets,
                        "sweeps": self.sweeps,
                        "known_obstacles": sorted(self.known_obstacles),
                        "known_resources": sorted(self.known_resources),
                        "resource_last_seen": {
                            f"{cell[0]},{cell[1]}": tick
                            for cell, tick in self.resource_last_seen.items()
                        },
                        "scout_seen": {
                            f"{cell[0]},{cell[1]}": tick
                            for cell, tick in self.scout_seen.items()
                        },
                        "scout_targets": {
                            worker_id: list(target)
                            for worker_id, target in self.scout_targets.items()
                        },
                        "scout_positions": {
                            worker_id: [list(position) for position in positions]
                            for worker_id, positions in self.scout_positions.items()
                        },
                        "position_stalls": self.position_stalls,
                        "resource_assignments": {
                            worker_id: list(target)
                            for worker_id, target in self.resource_assignments.items()
                        },
                        "last_move_destinations": {
                            worker_id: list(target)
                            for worker_id, target in self.last_move_destinations.items()
                        },
                        "depleted": {
                            f"{cell[0]},{cell[1]}": until
                            for cell, until in self.depleted.items()
                        },
                        "contested": {
                            f"{cell[0]},{cell[1]}": until
                            for cell, until in self.contested.items()
                        },
                        "resource_progress": {
                            worker_id: [
                                progress.target[0],
                                progress.target[1],
                                progress.best_cost,
                                progress.stalled_ticks,
                            ]
                            for worker_id, progress in self.resource_progress.items()
                        },
                        "core_threat_until_tick": self.core_threat_until_tick,
                        "core_intercept_worker_id": self.core_intercept_worker_id,
                        "economic_history": self.economic_history,
                    }
                )
            )
            temporary.replace(self.path)
        except OSError:
            log.warning("could not persist scout state to %s", self.path)
            return
        self.dirty = False
        if tick is not None:
            self.last_saved_tick = tick

    def forget_resource(self, cell: Position) -> None:
        """Discard one stale resource and any Worker assignment to it."""

        changed = cell in self.known_resources or cell in self.resource_last_seen
        self.known_resources.discard(cell)
        self.resource_last_seen.pop(cell, None)
        for worker_id, target in list(self.resource_assignments.items()):
            if target == cell:
                del self.resource_assignments[worker_id]
                changed = True
        if changed:
            self.dirty = True

    def record_economy(self, tick: int, flow: Counter) -> None:
        """Record one authoritative resolution sample for adaptive planning."""

        sample = (
            tick,
            flow["harvest"],
            flow["deposit"] + flow["capture"],
            flow["dropped"] + flow["overflow"],
        )
        if self.economic_history and self.economic_history[-1][0] == tick:
            if self.economic_history[-1] == sample:
                return
            self.economic_history[-1] = sample
        elif not self.economic_history or tick > self.economic_history[-1][0]:
            self.economic_history.append(sample)
        else:
            samples = {item[0]: item for item in self.economic_history}
            samples[tick] = sample
            self.economic_history = [samples[key] for key in sorted(samples)]
        del self.economic_history[:-ECONOMY_HISTORY_LIMIT]
        self.dirty = True

    def economic_totals(self, tick: int, window: int = ECONOMY_FLOW_WINDOW) -> Counter:
        """Sum recent harvest, income, and losses without inventing node yields."""

        totals: Counter = Counter()
        cutoff = tick - window + 1
        for sample_tick, harvested, income, lost in self.economic_history:
            if sample_tick < cutoff or sample_tick > tick:
                continue
            totals["harvest"] += harvested
            totals["income"] += income
            totals["lost"] += lost
            totals["samples"] += 1
        return totals

    def prune_workers(self, living_worker_ids: set[str]) -> None:
        """Remove per-Worker history after a Worker dies or a fleet respawns."""

        changed = False
        for mapping in (
            self.offsets,
            self.sweeps,
            self.scout_targets,
            self.scout_positions,
            self.position_stalls,
            self.resource_assignments,
            self.resource_progress,
            self.last_move_destinations,
        ):
            for worker_id in set(mapping) - living_worker_ids:
                del mapping[worker_id]
                changed = True
        self.recalling_workers.intersection_update(living_worker_ids)
        if (
            self.core_intercept_worker_id is not None
            and self.core_intercept_worker_id not in living_worker_ids
        ):
            self.core_intercept_worker_id = None
            changed = True
        for key in list(self.resource_cooldowns):
            if key[0] not in living_worker_ids:
                del self.resource_cooldowns[key]
        if changed:
            self.dirty = True

    def prune_obstacles(self, core_position: Position) -> int:
        """Bound permanent terrain memory by dropping the most distant cells.

        Obstacles never expire by age, so the only safe relevance signal is
        distance from the Core: anything beyond the farthest trip or scouting
        radius cannot change a route, and is cheaply re-learned by sight if the
        Core ever migrates back.  Returns the number of cells dropped.
        """

        if len(self.known_obstacles) <= OBSTACLE_MEMORY_MAX_CELLS:
            return 0
        ordered = sorted(
            self.known_obstacles,
            key=lambda cell: (manhattan(core_position, cell), cell[0], cell[1]),
        )
        dropped = ordered[OBSTACLE_MEMORY_KEEP_CELLS:]
        self.known_obstacles.difference_update(dropped)
        self.dirty = True
        return len(dropped)

    def expire(self, tick: int, living_worker_ids: set[str] | None = None) -> None:
        """Forget depleted, contested, and threatened cells after their cooldowns."""

        for cell in [c for c, until in self.depleted.items() if until <= tick]:
            del self.depleted[cell]
        for cell in [c for c, until in self.contested.items() if until <= tick]:
            del self.contested[cell]
        for cell in [c for c, until in self.threatened.items() if until <= tick]:
            del self.threatened[cell]
        for key in [key for key, until in self.resource_cooldowns.items() if until <= tick]:
            del self.resource_cooldowns[key]
        active_targets = {
            target
            for worker_id, target in self.resource_assignments.items()
            if living_worker_ids is None or worker_id in living_worker_ids
        }
        stale = []
        for cell, last_seen in self.resource_last_seen.items():
            age = tick - last_seen
            ttl = (
                RESOURCE_ACTIVE_ASSIGNMENT_TTL
                if cell in active_targets
                else RESOURCE_MEMORY_TTL
            )
            if age > ttl:
                stale.append(cell)
        for cell in stale:
            self.forget_resource(cell)
        old_seen = [
            cell
            for cell, last_seen in self.scout_seen.items()
            if tick - last_seen > SCOUT_COVERAGE_TTL
        ]
        for cell in old_seen:
            del self.scout_seen[cell]
            self.dirty = True
        overflow = len(self.scout_seen) - SCOUT_COVERAGE_MAX_CELLS
        if overflow > 0:
            oldest = sorted(
                self.scout_seen,
                key=lambda cell: (self.scout_seen[cell], cell[0], cell[1]),
            )[:overflow]
            for cell in oldest:
                del self.scout_seen[cell]
            self.dirty = True

    def record_position(self, worker_id: str, position: Position) -> None:
        history = self.scout_positions.setdefault(worker_id, [])
        if history and history[-1] == position:
            # Holding one cell adds no history, which used to make a frozen
            # Worker indistinguishable from one halfway along a healthy route.
            # Count the idle Ticks so is_looping can see a standstill.
            self.position_stalls[worker_id] = self.position_stalls.get(worker_id, 0) + 1
            self.dirty = True
            return
        history.append(position)
        del history[:-SCOUT_HISTORY_LIMIT]
        self.position_stalls.pop(worker_id, None)
        self.dirty = True

    def is_looping(self, worker_id: str) -> bool:
        if self.position_stalls.get(worker_id, 0) >= WORKER_STALL_TICKS:
            return True
        history = self.scout_positions.get(worker_id, [])
        if len(history) < SCOUT_LOOP_WINDOW:
            return False
        recent = history[-SCOUT_LOOP_WINDOW:]
        return len(set(recent)) <= 3 and manhattan(recent[0], recent[-1]) <= 2


@dataclass
class MovementReservations:
    """One-Tick ledger of cells claimed by already planned moves."""

    destinations: set[Position] = field(default_factory=set)
    by_unit: dict[UUID, Position] = field(default_factory=dict)

    def reserve(self, unit_id: UUID, destination: Position) -> None:
        self.destinations.add(destination)
        self.by_unit[unit_id] = destination


@dataclass
class TickBudget:
    """Resources and Core storage still unspent while planning one Tick."""

    resources: int
    space: int
    projected_deposits: int = 0


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
    # A known wall can make exact A* expensive even though the infinite map is
    # still navigable.  Preserve the best admissible frontier estimate so a
    # Worker is not sent scouting merely because the estimator hit its budget.
    if frontier:
        return min(estimate for estimate, _, _ in frontier)
    return None


def is_legal_shot(origin: Position, cell: Position, obstacles: frozenset[Position]) -> bool:
    dx = cell[0] - origin[0]
    dy = cell[1] - origin[1]
    if dx == 0 and dy == 0:
        return False
    if dx != 0 and dy != 0 and abs(dx) != abs(dy):
        return False
    distance = max(abs(dx), abs(dy))
    if not 1 <= distance <= 3:
        return False
    step_x = 1 if dx > 0 else (-1 if dx < 0 else 0)
    step_y = 1 if dy > 0 else (-1 if dy < 0 else 0)
    for i in range(1, distance):
        if (origin[0] + i * step_x, origin[1] + i * step_y) in obstacles:
            return False
    return True


def combat_enemies(
    enemies: tuple[CoreView | UnitView, ...],
) -> tuple[UnitView, ...]:
    """Return visible enemies that can actually damage the Core or Units."""

    return tuple(
        enemy
        for enemy in enemies
        if getattr(enemy, "unit_type", None) in {UnitType.VANGUARD, UnitType.RANGER}
    )


def projected_core_damage(
    core_position: Position,
    enemies: tuple[CoreView | UnitView, ...],
    obstacles: frozenset[Position],
) -> int:
    """Count attacks visible enemies can legally land on the Core this Tick."""

    damage = 0
    for enemy in combat_enemies(enemies):
        if enemy.unit_type is UnitType.VANGUARD:
            damage += int(manhattan(enemy.position, core_position) == 1)
        elif is_legal_shot(enemy.position, core_position, obstacles):
            damage += 1
    return damage


def projected_post_combat_capacity(
    turn,
    enemies: tuple[CoreView | UnitView, ...],
    obstacles: frozenset[Position],
) -> int:
    """Return a conservative storage cap after visible combat losses.

    Enemy actions are private, so exact casualties are unknowable. Treat every
    currently killable friendly Unit as a possible loss. This can reserve more
    cautiously than settlement, but it prevents deposits or Core actions from
    relying on resources the server may destroy before healing and spawning.
    """

    hostile_units = combat_enemies(enemies)
    possible_loss_costs: list[int] = []
    for unit in turn.units:
        # Movement resolves before combat. Include the current cell because a
        # contested or blocked move may leave the Unit there, and include the
        # planned destination when the plan already contains a MOVE. Before
        # Unit planning this naturally evaluates only current positions.
        possible_positions = {unit.position, projected_unit_position(unit, turn.plan)}
        incoming = 0
        for enemy in hostile_units:
            if enemy.unit_type is UnitType.VANGUARD:
                incoming += int(
                    any(
                        manhattan(enemy.position, position) == 1
                        for position in possible_positions
                    )
                )
            elif any(
                is_legal_shot(enemy.position, position, obstacles)
                for position in possible_positions
            ):
                incoming += 1
        if incoming >= unit.hp:
            possible_loss_costs.append(unit.hp)
    # One Ranger can damage one object. A Vanguard sweep can damage every Unit
    # in one adjacent cell; two is the normal capacity, while the occupancy
    # count also covers any historical over-capacity state supplied by a Turn.
    unit_occupancy = Counter(unit.position for unit in turn.units)
    attack_budget = sum(
        (
            max(
                2,
                *(unit_occupancy[
                    (
                        enemy.position[0] + dx,
                        enemy.position[1] + dy,
                    )
                ] for dx, dy in DIRECTION_DELTAS.values()),
            )
            if enemy.unit_type is UnitType.VANGUARD
            else 1
        )
        for enemy in hostile_units
    )
    possible_losses = 0
    for hp in sorted(possible_loss_costs):
        if hp > attack_budget:
            break
        attack_budget -= hp
        possible_losses += 1
    population = max(0, turn.state.population - possible_losses)
    return core_resource_capacity(population)


def core_threatening_enemies(
    core_position: Position,
    enemies: tuple[CoreView | UnitView, ...],
    obstacles: frozenset[Position],
) -> tuple[UnitView, ...]:
    """Return enemies able to damage the Core from their current cells."""

    return tuple(
        enemy
        for enemy in combat_enemies(enemies)
        if (
            enemy.unit_type is UnitType.VANGUARD
            and manhattan(enemy.position, core_position) == 1
        )
        or (
            enemy.unit_type is UnitType.RANGER
            and is_legal_shot(enemy.position, core_position, obstacles)
        )
    )


def core_in_danger(
    core_position: Position,
    enemies: tuple[CoreView | UnitView, ...],
    obstacles: frozenset[Position],
    distance: int,
) -> bool:
    """Return whether the Core is under legal fire or near a combat Unit."""

    hostile_units = combat_enemies(enemies)
    return bool(core_threatening_enemies(core_position, enemies, obstacles)) or any(
        manhattan(core_position, enemy.position) <= distance
        for enemy in hostile_units
    )


def player_holds_beacon(turn) -> bool:
    beacon = turn.beacon
    if beacon is None or beacon.status is not BeaconStatus.CARRIED:
        return False
    owned_ids = {unit.id for unit in turn.units}
    if turn.core is not None:
        owned_ids.add(turn.core.id)
    return beacon.carrier_id in owned_ids


def core_shield_target(turn) -> int:
    return 10 if player_holds_beacon(turn) else CORE_SHIELD_FLOOR


def blocked_cells(
    units: tuple,
    core_position: Position,
    obstacles: frozenset[Position],
    enemy_cells: set[Position],
) -> set[Position]:
    """Return every cell no owned object may step into this Tick.

    Obstacles and enemies are always impassable. A friendly cell is impassable
    only once two objects already stand on it, so a single Unit never walls off
    the army behind it.
    """

    occupancy = Counter(unit.position for unit in units)
    occupancy[core_position] += 1
    crowded = {cell for cell, count in occupancy.items() if count >= 2}
    return set(obstacles) | enemy_cells | crowded


def enemy_threat_cells(
    enemies: tuple[CoreView | UnitView, ...],
    obstacles: frozenset[Position],
) -> set[Position]:
    """Return cells a visible Vanguard or Ranger can damage this Tick."""

    threatened: set[Position] = set()
    for enemy in enemies:
        unit_type = getattr(enemy, "unit_type", None)
        if unit_type is UnitType.VANGUARD:
            for direction in DIRECTION_ORDER:
                dx, dy = DIRECTION_DELTAS[direction]
                threatened.add((enemy.position[0] + dx, enemy.position[1] + dy))
        elif unit_type is UnitType.RANGER:
            for dx, dy in SCOUT_OFFSETS:
                for distance in range(1, 4):
                    cell = (
                        enemy.position[0] + dx * distance,
                        enemy.position[1] + dy * distance,
                    )
                    if cell in obstacles:
                        break
                    threatened.add(cell)
    return threatened


def remember_threat_cells(
    turn,
    memory: ScoutMemory,
    obstacles: frozenset[Position],
) -> set[Position]:
    """Return recently threatened cells, not only this Tick's visible arcs.

    An enemy slips in and out of vision as either side moves, so rebuilding the
    avoid-set from scratch every Tick reverses the route every Tick: a loaded
    Worker steps into a threatened cell, sees the shooter, retreats, loses sight
    of it, and walks back in.  Keeping each arc for ``THREAT_MEMORY_TICKS`` makes
    the detour stable enough to actually walk around the threat.
    """

    for cell in enemy_threat_cells(turn.visible_enemies, obstacles):
        memory.threatened[cell] = turn.tick + THREAT_MEMORY_TICKS
    return set(memory.threatened)


def step_toward(
    start: Position,
    goal: Position,
    blocked: set[Position],
    reserved: set[Position] = frozenset(),
) -> Position | None:
    """Return the first deterministic pathfinding step toward goal.

    Open-map scout routes use a cheap Manhattan fast path, which stays O(distance)
    and works for arbitrarily distant waypoints. A* is reserved for routes that
    need to navigate around known blocked cells.
    """

    if start == goal:
        return None
    if goal in reserved:
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

    # f-score, remaining distance, g-score, discovery order, cell, first step
    frontier: list[tuple[int, int, int, int, Position, Position | None]] = []
    heappush(frontier, (manhattan(start, goal), manhattan(start, goal), 0, 0, start, None))
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
        # A distant route around known walls may exceed one Tick's search
        # budget.  Follow the best partial route and replan from fresh state on
        # the next Tick instead of waiting forever for a complete path.
        _, _, _, _, _, first_step = min(
            frontier,
            key=lambda item: (item[1], item[2], item[3]),
        )
        return first_step
    return None


def move_or_wait(
    unit,
    goal: Position,
    blocked: set[Position],
    reservations: MovementReservations | None = None,
) -> bool:
    """Queue a non-conflicting step toward goal, or wait when none exists."""

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
    reservations: MovementReservations | None = None,
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


def _line_cells(start: Position, end: Position) -> tuple[Position, ...]:
    """Return the integer supercover line, including exact corner touches."""

    x, y = start
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    nx, ny = abs(dx), abs(dy)
    sx = 1 if dx > 0 else -1 if dx < 0 else 0
    sy = 1 if dy > 0 else -1 if dy < 0 else 0
    ix = iy = 0
    cells: list[Position] = [(x, y)]

    def append(cell: Position) -> None:
        if cells[-1] != cell:
            cells.append(cell)

    while ix < nx or iy < ny:
        x_cross = (1 + 2 * ix) * ny
        y_cross = (1 + 2 * iy) * nx
        if x_cross == y_cross:
            append((x + sx, y))
            append((x, y + sy))
            x += sx
            y += sy
            ix += 1
            iy += 1
            append((x, y))
        elif x_cross < y_cross:
            x += sx
            ix += 1
            append((x, y))
        else:
            y += sy
            iy += 1
            append((x, y))
    return tuple(cells)


def cell_visible_to_friendly(turn, target: Position) -> bool:
    """Return exact current visibility from the Core and controlled Units."""

    obstacles = turn.obstacle_cells
    observers = [(turn.core.position, CORE_VISION_RADIUS)] if turn.core else []
    observers.extend(
        (unit.position, UNIT_VISION_RADII[unit.unit_type])
        for unit in turn.units
        if unit.unit_type in UNIT_VISION_RADII
    )
    return _cell_visible(target, obstacles, observers)


def _cell_visible(
    target: Position,
    obstacles: frozenset[Position] | set[Position],
    observers: list[tuple[Position, int]],
) -> bool:
    """Visibility primitive used by both one-off checks and coverage scans."""

    for origin, radius in observers:
        if manhattan(origin, target) > radius:
            continue
        if not any(cell in obstacles for cell in _line_cells(origin, target)[1:-1]):
            return True
    return False


def scout_grid_key(
    position: Position,
    core_position: Position | None = None,
) -> tuple[int, int]:
    """Map a world position to a fixed-origin coarse grid cell.

    ``core_position`` remains an accepted, ignored argument for callers of the
    previous helper.  Grid keys are now absolute world coordinates, so Core
    migration cannot reinterpret old coverage records.
    """

    del core_position
    return (
        position[0] // SCOUT_GRID_SIZE,
        position[1] // SCOUT_GRID_SIZE,
    )


def scout_grid_center(
    cell: tuple[int, int],
    core_position: Position | None = None,
) -> Position:
    """Return the fixed-origin center of an absolute scout grid cell."""

    del core_position
    gx, gy = cell
    half = SCOUT_GRID_SIZE // 2
    return (gx * SCOUT_GRID_SIZE + half, gy * SCOUT_GRID_SIZE + half)


def chunk_resource_quota(position: Position) -> int:
    """Return the documented natural-node quota for a world chunk."""

    chunk_x, chunk_y = position[0] // 32, position[1] // 32

    def axis(value: int) -> int:
        return value if value >= 0 else -value - 1

    ring = axis(chunk_x) + axis(chunk_y)
    return max(2, (16 * 8) // (8 + ring))


def mark_scout_coverage(turn, memory: ScoutMemory) -> None:
    """Mark coarse cells currently covered by the Core and controlled Units."""

    if turn.core is None:
        return
    observers = [(turn.core.position, CORE_VISION_RADIUS)] + [
        (unit.position, UNIT_VISION_RADII[unit.unit_type])
        for unit in turn.units
        if unit.unit_type in UNIT_VISION_RADII
    ]
    obstacles = turn.obstacle_cells
    for origin, radius in observers:
        span = radius // SCOUT_GRID_SIZE + 2
        base = scout_grid_key(origin)
        for gx in range(base[0] - span, base[0] + span + 1):
            for gy in range(base[1] - span, base[1] + span + 1):
                cell = (gx, gy)
                center = scout_grid_center(cell)
                if _cell_visible(center, obstacles, observers):
                    if memory.scout_seen.get((gx, gy), -1) != turn.tick:
                        memory.scout_seen[(gx, gy)] = turn.tick
                        memory.dirty = True


def scout_grid_disc(core_position: Position, max_distance: int) -> list[tuple[int, int]]:
    """Return the absolute grid cells whose centers lie inside the scout disc."""

    center_cell = scout_grid_key(core_position)
    grid_radius = max_distance // SCOUT_GRID_SIZE + 2
    return [
        (gx, gy)
        for gx in range(center_cell[0] - grid_radius, center_cell[0] + grid_radius + 1)
        for gy in range(center_cell[1] - grid_radius, center_cell[1] + grid_radius + 1)
        if manhattan(core_position, scout_grid_center((gx, gy))) <= max_distance
    ]


def scout_disc_radius(core_position: Position, memory: ScoutMemory) -> int:
    """Return the search radius, widened only once the near disc is explored."""

    disc = scout_grid_disc(core_position, SCOUT_MAX_DISTANCE)
    if not disc:
        return SCOUT_MAX_DISTANCE
    seen = sum(1 for cell in disc if cell in memory.scout_seen)
    if seen / len(disc) >= SCOUT_EXHAUSTED_RATIO:
        return SCOUT_FAR_DISTANCE
    return SCOUT_MAX_DISTANCE


def scout_coverage_target(
    worker,
    core_position: Position,
    memory: ScoutMemory,
    claimed: set[tuple[int, int]],
    tick: int,
    blocked: set[Position] | frozenset[Position] = frozenset(),
    max_distance: int = SCOUT_MAX_DISTANCE,
) -> Position | None:
    """Select a stable absolute grid target within the current scout disc."""

    worker_id = str(worker.id)
    existing = memory.scout_targets.get(worker_id)
    if existing is not None and existing not in claimed:
        target = scout_grid_center(existing)
        if manhattan(core_position, target) > max_distance or target in blocked:
            log.info(
                "tick=%d worker %s dropping out-of-range scout target=%s",
                tick,
                worker_id[:8],
                target,
            )
            memory.scout_targets.pop(worker_id, None)
            memory.dirty = True
        elif (
            memory.scout_seen.get(existing) == tick
            or manhattan(worker.position, target) <= SCOUT_ARRIVAL_DISTANCE
        ):
            memory.scout_seen[existing] = tick
            memory.scout_targets.pop(worker_id, None)
            memory.dirty = True
        elif not memory.is_looping(worker_id):
            claimed.add(existing)
            return target
        else:
            log.info(
                "tick=%d worker %s releasing looping scout cell=%s",
                tick,
                worker_id[:8],
                existing,
            )
            memory.scout_targets.pop(worker_id, None)
            memory.scout_seen[existing] = tick
            memory.dirty = True

    center_cell = scout_grid_key(core_position)
    candidate_cells = scout_grid_disc(core_position, max_distance)

    candidates: list[tuple[int, int, int, int, int, int, tuple[int, int]]] = []
    for gx, gy in candidate_cells:
        cell = (gx, gy)
        if cell in claimed:
            continue
        target = scout_grid_center(cell)
        if target in blocked:
            continue
        distance = manhattan(worker.position, target)
        if distance == 0:
            continue
        last_seen = memory.scout_seen.get(cell, -1)
        age = tick - last_seen if last_seen >= 0 else SCOUT_COVERAGE_TTL
        candidates.append(
            (
                0 if last_seen < 0 else 1,
                -age,
                -chunk_resource_quota(target),
                distance,
                abs(gx - center_cell[0]) + abs(gy - center_cell[1]),
                gx,
                (gx, gy),
            )
        )
    if not candidates:
        return None
    _, _, _, _, _, _, cell = min(candidates)
    memory.scout_targets[worker_id] = cell
    claimed.add(cell)
    memory.dirty = True
    log.debug("tick=%d scout coverage worker=%s cell=%s", tick, worker_id[:8], cell)
    return scout_grid_center(cell)


def remember_obstacles(turn, memory: ScoutMemory) -> None:
    """Keep observed obstacles, bounded by distance, because terrain is permanent."""

    discovered = set(turn.obstacle_cells) - memory.known_obstacles
    if not discovered:
        return
    memory.known_obstacles.update(discovered)
    memory.dirty = True
    dropped = memory.prune_obstacles(turn.core.position) if turn.core else 0
    if dropped:
        log.info(
            "tick=%d pruned distant obstacles=%d total=%d",
            turn.tick,
            dropped,
            len(memory.known_obstacles),
        )
    log.debug(
        "tick=%d learned obstacles=%d total=%d",
        turn.tick,
        len(discovered),
        len(memory.known_obstacles),
    )


def remember_resources(turn, memory: ScoutMemory) -> None:
    """Merge sightings and invalidate only resources currently visible as gone."""

    # The current Turn is authoritative.  RESOURCE_DEPLETED can describe a
    # same-cell loser while a partially harvested cargo pile remains.
    revived = set(turn.resource_cells) & set(memory.depleted)
    for cell in revived:
        memory.depleted.pop(cell, None)
    visible = set(turn.resource_cells)
    new_cells = visible - memory.known_resources
    if new_cells:
        memory.known_resources.update(new_cells)
        memory.dirty = True
        log.info("tick=%d discovered resources=%s", turn.tick, sorted(new_cells))
    for cell in visible:
        if memory.resource_last_seen.get(cell) != turn.tick:
            memory.resource_last_seen[cell] = turn.tick
            memory.dirty = True
    for cell in list(memory.known_resources):
        if cell in visible or cell in memory.depleted:
            continue
        if cell_visible_to_friendly(turn, cell):
            memory.forget_resource(cell)
            log.info("tick=%d forgetting visible stale resource=%s", turn.tick, cell)


def available_resources(turn, memory: ScoutMemory) -> set[Position]:
    """Return visible or remembered resource nodes that are not cooling down."""

    return (set(turn.resource_cells) | memory.known_resources) - set(memory.depleted)


def minimum_cost_assignment(costs: list[list[int]]) -> tuple[int, ...]:
    """Return one deterministic minimum-cost column for each matrix row."""

    if not costs:
        return ()
    row_count = len(costs)
    column_count = len(costs[0])
    if column_count < row_count or any(len(row) != column_count for row in costs):
        raise ValueError("assignment matrix must be rectangular with rows <= columns")

    row_potential = [0] * (row_count + 1)
    column_potential = [0] * (column_count + 1)
    matched_row = [0] * (column_count + 1)
    previous_column = [0] * (column_count + 1)

    for row_index in range(1, row_count + 1):
        matched_row[0] = row_index
        current_column = 0
        minimum_slack = [sys.maxsize] * (column_count + 1)
        visited = [False] * (column_count + 1)
        while True:
            visited[current_column] = True
            current_row = matched_row[current_column]
            delta = sys.maxsize
            next_column = 0
            for column_index in range(1, column_count + 1):
                if visited[column_index]:
                    continue
                reduced_cost = (
                    costs[current_row - 1][column_index - 1]
                    - row_potential[current_row]
                    - column_potential[column_index]
                )
                if reduced_cost < minimum_slack[column_index]:
                    minimum_slack[column_index] = reduced_cost
                    previous_column[column_index] = current_column
                if minimum_slack[column_index] < delta:
                    delta = minimum_slack[column_index]
                    next_column = column_index
            for column_index in range(column_count + 1):
                if visited[column_index]:
                    row_potential[matched_row[column_index]] += delta
                    column_potential[column_index] -= delta
                else:
                    minimum_slack[column_index] -= delta
            current_column = next_column
            if matched_row[current_column] == 0:
                break
        while True:
            next_column = previous_column[current_column]
            matched_row[current_column] = matched_row[next_column]
            current_column = next_column
            if current_column == 0:
                break

    assignment = [-1] * row_count
    for column_index in range(1, column_count + 1):
        row_index = matched_row[column_index]
        if row_index:
            assignment[row_index - 1] = column_index - 1
    return tuple(assignment)


def refresh_resource_progress(
    workers,
    resources: set[Position],
    blocked: set[Position] | frozenset[Position],
    tick: int,
    memory: ScoutMemory,
) -> None:
    """Release a resource intent after six consecutive Ticks without progress."""

    worker_by_id = {str(worker.id): worker for worker in workers}
    for worker_id in list(memory.resource_progress):
        target = memory.resource_assignments.get(worker_id)
        if worker_id not in worker_by_id or target not in resources:
            memory.resource_progress.pop(worker_id, None)

    for worker_id, target in list(memory.resource_assignments.items()):
        worker = worker_by_id.get(worker_id)
        if worker is None or target not in resources:
            memory.resource_progress.pop(worker_id, None)
            continue
        cost = None
        if target not in blocked or worker.position == target:
            cost = bounded_route_cost(worker.position, target, blocked)
        measured = cost if cost is not None else PATH_COST_UNREACHABLE
        progress = memory.resource_progress.get(worker_id)
        if progress is None or progress.target != target:
            memory.resource_progress[worker_id] = ResourceProgress(target, measured)
            continue
        if measured < progress.best_cost:
            progress.best_cost = measured
            progress.stalled_ticks = 0
            continue
        progress.stalled_ticks += 1
        if progress.stalled_ticks < RESOURCE_STALL_TICKS:
            continue
        memory.resource_cooldowns[(worker_id, target)] = tick + RESOURCE_COOLDOWN_TICKS
        memory.resource_assignments.pop(worker_id, None)
        memory.resource_progress.pop(worker_id, None)
        memory.dirty = True
        log.info(
            "tick=%d worker %s releasing stalled resource=%s",
            tick,
            worker_id[:8],
            target,
        )


def assign_resource_targets(
    workers,
    resources: set[Position],
    blocked: set[Position] | frozenset[Position] = frozenset(),
    previous: dict[str, Position] | None = None,
    max_cost: int | None = RESOURCE_MAX_ASSIGNMENT_COST,
    depot: Position | None = None,
    *,
    max_total_cost: int | None = None,
    tick: int | None = None,
    last_seen: dict[Position, int] | None = None,
    cooldowns: dict[tuple[str, Position], int] | None = None,
    remote_distance: int | None = None,
    max_remote_workers: int | None = None,
) -> dict[UUID, Position]:
    """Pair Workers to distinct reachable nodes using minimum-cost matching.

    An empty Worker already standing on a current node claims it first.  Other
    routes must satisfy the configured leg and complete round-trip limits.
    Stale sightings are mildly penalized and an existing intent gets a small
    bonus, reducing churn without committing the fleet to uneconomic trips.
    """

    previous = previous or {}
    cooldowns = cooldowns or {}
    ordered_workers = sorted(workers, key=lambda worker: str(worker.id))
    ordered_resources = sorted(resources)
    targets: dict[UUID, Position] = {}
    claimed: set[Position] = set()

    # Harvesting in place is always the cheapest useful action and must happen
    # before any optional economic cutoff supplied by a helper caller.
    for cell in ordered_resources:
        candidates = [worker for worker in ordered_workers if worker.position == cell]
        if not candidates:
            continue
        worker = candidates[0]
        targets[worker.id] = cell
        claimed.add(cell)

    ordered_workers = [worker for worker in ordered_workers if worker.id not in targets]
    ordered_resources = [cell for cell in ordered_resources if cell not in claimed]
    if not ordered_workers or not ordered_resources:
        return targets

    return_costs: dict[Position, int | None] = {}
    if depot is not None:
        return_costs = {
            cell: bounded_route_cost(cell, depot, blocked) for cell in ordered_resources
        }

    unassigned_cost = PATH_COST_UNREACHABLE * (len(ordered_workers) + 1)
    forbidden_cost = unassigned_cost * 2
    cost_matrix: list[list[int]] = []
    for worker in ordered_workers:
        worker_id = str(worker.id)
        row: list[int] = []
        for cell in ordered_resources:
            if tick is not None and cooldowns.get((worker_id, cell), 0) > tick:
                row.append(forbidden_cost)
                continue
            if cell in blocked and worker.position != cell:
                row.append(forbidden_cost)
                continue
            cost = bounded_route_cost(worker.position, cell, blocked)
            if cost is None or (max_cost is not None and cost > max_cost):
                row.append(forbidden_cost)
                continue
            return_cost = return_costs.get(cell, 0)
            if max_cost is not None and (return_cost is None or return_cost > max_cost):
                row.append(forbidden_cost)
                continue
            if return_cost is None:
                return_cost = manhattan(cell, depot) if depot is not None else 0
            total_cost = cost + return_cost
            if max_total_cost is not None and total_cost > max_total_cost:
                row.append(forbidden_cost)
                continue
            age = 0
            if tick is not None and last_seen is not None:
                age = max(0, tick - last_seen.get(cell, tick))
            stale_penalty = 0 if age == 0 else min(
                RESOURCE_STALE_PENALTY_MAX,
                2 + age // 8,
            )
            adjusted = total_cost + stale_penalty
            if previous.get(worker_id) == cell:
                adjusted = max(0, adjusted - RESOURCE_REASSIGN_BONUS)
            row.append(adjusted)
        row.extend([unassigned_cost] * len(ordered_workers))
        cost_matrix.append(row)

    for row_index, column_index in enumerate(minimum_cost_assignment(cost_matrix)):
        if column_index >= len(ordered_resources):
            continue
        if cost_matrix[row_index][column_index] >= forbidden_cost:
            continue
        targets[ordered_workers[row_index].id] = ordered_resources[column_index]

    if (
        depot is not None
        and remote_distance is not None
        and max_remote_workers is not None
    ):
        worker_by_id = {worker.id: worker for worker in workers}
        remote = []
        for worker_id, cell in targets.items():
            worker = worker_by_id[worker_id]
            if worker.position == cell:
                continue
            return_cost = return_costs.get(cell)
            if return_cost is None or return_cost > remote_distance:
                outbound = bounded_route_cost(worker.position, cell, blocked)
                remote.append(
                    (
                        return_cost if return_cost is not None else PATH_COST_UNREACHABLE,
                        outbound if outbound is not None else PATH_COST_UNREACHABLE,
                        str(worker_id),
                        worker_id,
                    )
                )
        for _, _, _, worker_id in sorted(remote)[max(0, max_remote_workers) :]:
            targets.pop(worker_id, None)
    return targets


def scout_phase(worker_id: str) -> int:
    """Return a stable per-Worker starting ray, spreading Workers apart.

    Uses an explicit byte sum rather than hash() because the built-in is salted
    per process and would reshuffle every restart.
    """

    return sum(worker_id.encode()) % len(SCOUT_OFFSETS)


def scout_waypoint(step: int, phase: int) -> tuple[int, int, int]:
    """Return direction and ring for one bounded, repeating scout step."""

    layer_size = len(SCOUT_OFFSETS) * SCOUT_RING_COUNT
    ray_step, position = divmod(step % layer_size, SCOUT_RING_COUNT)
    _, ray = divmod(ray_step, len(SCOUT_OFFSETS))
    ring = position if ray % 2 == 0 else SCOUT_RING_COUNT - 1 - position
    direction = (phase + ray) % len(SCOUT_OFFSETS)
    return direction, ring, 0


def scout_target(worker_id: str, core_position: Position, memory: ScoutMemory) -> Position:
    """Return a bounded Core-relative fallback waypoint."""

    step = memory.offsets.get(worker_id, 0)
    direction, ring, _ = scout_waypoint(step, scout_phase(worker_id))
    radius = min(SCOUT_MAX_DISTANCE, SCOUT_MIN_RADIUS + ring * SCOUT_RADIUS_STEP)
    dx, dy = SCOUT_OFFSETS[direction]
    scale = radius // (abs(dx) + abs(dy))
    return (core_position[0] + dx * scale, core_position[1] + dy * scale)


def advance_scout(worker_id: str, memory: ScoutMemory) -> None:
    """Move this Worker to the next waypoint, wrapping within the safe ring."""

    previous = memory.offsets.get(worker_id, 0)
    layer_size = len(SCOUT_OFFSETS) * SCOUT_RING_COUNT
    nxt = (previous + 1) % layer_size
    if nxt == 0:
        memory.sweeps[worker_id] = memory.sweeps.get(worker_id, 0) + 1
        log.info(
            "worker %s finished bounded scout sweep #%d out to radius %d",
            worker_id[:8],
            memory.sweeps[worker_id],
            SCOUT_MAX_DISTANCE,
        )
    memory.offsets[worker_id] = nxt
    memory.dirty = True


def is_failure(event_type: str) -> bool:
    return any(marker in event_type for marker in FAILURE_MARKERS)


def observe(turn, memory: ScoutMemory) -> Counter:
    """Consume last Tick's resolution events into logs and memory."""

    counts: Counter = Counter()
    flow: Counter = Counter()
    for event in turn.events:
        counts[event.event_type] += 1
        amount = getattr(event, "resource_amount", None) or 0
        if amount > 0:
            if event.event_type == "HARVEST_SUCCEEDED":
                flow["harvest"] += amount
            elif event.event_type == "DEPOSIT_SUCCEEDED":
                flow["deposit"] += amount
            elif event.event_type == "CORE_RESOURCES_CAPTURED":
                flow["capture"] += amount
            elif event.event_type == "WORKER_CARGO_DROPPED":
                flow["dropped"] += amount
            elif event.event_type == "CORE_RESOURCE_OVERFLOW_DESTROYED":
                flow["overflow"] += amount
            else:
                flow["other"] += amount
        if is_failure(event.event_type):
            log.warning(
                "tick=%d rejected event=%s reason=%s actor=%s position=%s values=%s",
                turn.tick,
                event.event_type,
                event.reason_code,
                str(event.actor_id)[:8] if event.actor_id is not None else None,
                event.position,
                event.values,
            )
            if (
                event.event_type == "HARVEST_FAILED"
                and event.reason_code in STALE_RESOURCE_REASONS
                and event.position is not None
            ):
                if event.position in turn.resource_cells:
                    memory.depleted.pop(event.position, None)
                    memory.known_resources.add(event.position)
                    memory.resource_last_seen[event.position] = turn.tick
                    memory.dirty = True
                else:
                    memory.depleted[event.position] = turn.tick + DEPLETED_TTL
                    memory.forget_resource(event.position)
            if event.reason_code == "MOVE_CONTESTED" and event.actor_id is not None:
                # UNIT_MOVE_FAILED.position is the unchanged origin.  Avoid the
                # destination we actually queued on the previous Tick instead.
                destination = memory.last_move_destinations.get(str(event.actor_id))
                if destination is not None:
                    memory.contested[destination] = turn.tick + CONTESTED_CELL_TTL
                    log.info(
                        "tick=%d avoiding contested destination=%s actor=%s",
                        turn.tick,
                        destination,
                        str(event.actor_id)[:8],
                    )
        else:
            log.debug("tick=%d event=%s values=%s", turn.tick, event.event_type, event.values)
        if event.event_type == "CORE_DAMAGED":
            previous_caution = memory.core_threat_until_tick
            memory.core_threat_until_tick = max(
                previous_caution,
                turn.tick + CORE_THREAT_CAUTION_TICKS,
            )
            memory.dirty |= memory.core_threat_until_tick != previous_caution
            log.warning(
                "tick=%d core damaged position=%s values=%s caution_until=%d",
                turn.tick,
                event.position,
                event.values,
                memory.core_threat_until_tick,
            )
        elif event.event_type == "CORE_DESTROYED":
            log.error(
                "tick=%d core destroyed reason=%s position=%s values=%s",
                turn.tick,
                event.reason_code,
                event.position,
                event.values,
            )
        elif event.event_type == "CORE_RESPAWNED":
            if memory.core_threat_until_tick:
                memory.core_threat_until_tick = 0
                memory.dirty = True
            log.warning(
                "tick=%d core respawned position=%s values=%s",
                turn.tick,
                event.position,
                event.values,
            )
        elif event.event_type == "CORE_SPAWN_SUCCEEDED":
            log.info(
                "tick=%d core spawn succeeded values=%s",
                turn.tick,
                event.values,
            )
    living_worker_ids = {str(worker.id) for worker in turn.workers}
    memory.prune_workers(living_worker_ids)
    memory.expire(turn.tick, living_worker_ids)
    memory.last_events = counts
    memory.last_resource_flow = flow
    memory.record_economy(turn.tick, flow)
    return counts


def affordable(cost: int, budget: TickBudget, capacity: int) -> bool:
    """Return whether a purchase fits the budget while keeping the heal reserve.

    At a low population the Core's capacity can be smaller than cost plus the
    reserve, which would make the purchase permanently impossible; the reserve
    is dropped in that case rather than deadlocking.
    """

    required = cost + CORE_HEAL_RESERVE
    if required > capacity:
        required = cost
    return budget.resources >= required


def economy_is_starving(turn, memory: ScoutMemory) -> bool:
    """Return whether a full measured window delivered nothing at all.

    This is deliberately income-only: what makes a region hopeless is that no
    resource reached the Core, not that a node was briefly visible.
    """

    totals = memory.economic_totals(turn.tick)
    return totals["samples"] >= STARVATION_SAMPLES and totals["income"] == 0


def core_should_escape_desert(turn, memory: ScoutMemory) -> bool:
    """Return whether the Core sits in a barren chunk with nothing to show.

    A low-quota outer chunk holds a fraction of the nodes the central area does.
    Chasing carriers around it keeps the base pinned exactly where the search has
    already failed, so a drought with no node in memory has to be walked out of.
    """

    core = turn.core
    if core is None or memory.known_resources or turn.resource_cells:
        return False
    if chunk_resource_quota(core.position) >= CORE_PREFERRED_RESOURCE_QUOTA:
        return False
    return economy_is_starving(turn, memory)


def resource_round_trip_budget(turn, memory: ScoutMemory) -> int:
    """Adapt expedition length to measured income without guessing pile size.

    A stalled or underfunded economy tightens to nearby nodes so it cannot
    repeat a fleet-wide long haul.  Once recent deliveries demonstrate healthy
    throughput, the bounded remote expeditions may range farther.
    """

    totals = memory.economic_totals(turn.tick)
    worker_price = unit_cost(UnitType.WORKER, turn.state.population)
    enough_history = totals["samples"] >= min(16, ECONOMY_FLOW_WINDOW)
    if turn.resources < worker_price or (enough_history and totals["income"] == 0):
        return RESOURCE_TRIP_COST_RECOVERY
    healthy_income = max(2, len(turn.workers) // 2)
    if totals["income"] >= healthy_income and totals["lost"] == 0:
        return RESOURCE_TRIP_COST_HEALTHY
    return RESOURCE_TRIP_COST_NORMAL


def remote_worker_limit(
    workers,
    resources: set[Position],
    depot: Position,
    blocked: set[Position],
    hostile_units: tuple[UnitView, ...],
    defense_caution: bool,
) -> int:
    """Allow a second remote expedition only for a safe, established fleet."""

    worker_count = len(workers)
    limit = max(1, worker_count // RESOURCE_REMOTE_WORKERS_PER_FLEET)
    if (
        worker_count < RESOURCE_ADAPTIVE_REMOTE_MIN_WORKERS
        or hostile_units
        or defense_caution
    ):
        return limit
    remote_resources = 0
    for cell in resources:
        cost = bounded_route_cost(cell, depot, blocked)
        if cost is None or cost <= RESOURCE_LOCAL_RETURN_DISTANCE:
            continue
        remote_resources += 1
        if remote_resources >= RESOURCE_ADAPTIVE_REMOTE_LIMIT:
            break
    if remote_resources >= RESOURCE_ADAPTIVE_REMOTE_LIMIT:
        limit = max(limit, RESOURCE_ADAPTIVE_REMOTE_LIMIT)
    return min(limit, max(1, worker_count // 2))


def core_is_stationary(turn) -> bool:
    return turn.core is not None and turn.core.view.state is CoreState.NORMAL


def projected_unit_position(unit, plan) -> Position:
    action = plan.unit_actions.get(unit.id)
    if not isinstance(action, MoveAction):
        return unit.position
    dx, dy = DIRECTION_DELTAS[action.direction]
    return (unit.position[0] + dx, unit.position[1] + dy)


def planned_core_departure_is_reliable(turn, unit) -> bool:
    """Return whether a planned move off the Core has no visible failure path."""

    action = turn.plan.unit_actions.get(unit.id)
    if not isinstance(action, MoveAction):
        return False
    destination = projected_unit_position(unit, turn.plan)
    if destination in turn.obstacle_cells:
        return False

    # Count current occupants even when they plan to leave: if their departure
    # fails, this move must still fit. Count all other planned arrivals too.
    current_occupants = sum(
        other.id != unit.id and other.position == destination for other in turn.units
    )
    if turn.core is not None and turn.core.position == destination:
        current_occupants += 1
    planned_arrivals = sum(
        other.id != unit.id
        and other.position != destination
        and projected_unit_position(other, turn.plan) == destination
        for other in turn.units
    )
    if current_occupants + planned_arrivals + 1 > 2:
        return False

    for enemy in turn.visible_enemies:
        if enemy.position == destination:
            return False
        if isinstance(enemy, UnitView) and manhattan(enemy.position, destination) == 1:
            return False
        if (
            isinstance(enemy, CoreView)
            and enemy.state is CoreState.MOVING
            and enemy.destination == destination
            and (enemy.move_progress or 0) + 1 >= (enemy.move_required_ticks or 4)
        ):
            return False
    return True


def spawn_cell_open(turn) -> bool:
    """Return whether the Core cell will reliably have one free spawn slot."""

    if turn.core is None or not core_is_stationary(turn):
        return False
    core_position = turn.core.position
    for unit in turn.units:
        if projected_unit_position(unit, turn.plan) == core_position:
            return False
        if unit.position == core_position and not planned_core_departure_is_reliable(
            turn,
            unit,
        ):
            return False
    return True


def desired_spawn_order(
    turn,
    enemies: tuple[CoreView | UnitView, ...],
    defense_caution: bool = False,
) -> tuple[UnitType, ...]:
    """Return a staged, adaptive production preference for this Turn."""

    workers = len(turn.workers)
    vanguards = len(turn.vanguards)
    rangers = len(turn.rangers)
    combat_units = vanguards + rangers
    hostile_units = combat_enemies(enemies)
    nearest_distance = min(
        (manhattan(turn.core.position, enemy.position) for enemy in hostile_units),
        default=None,
    )
    if nearest_distance is not None and nearest_distance <= DEFENDER_SPAWN_DISTANCE:
        preferred = UnitType.VANGUARD if nearest_distance <= 1 else UnitType.RANGER
        alternate = (
            UnitType.RANGER if preferred is UnitType.VANGUARD else UnitType.VANGUARD
        )
        return (preferred, alternate)

    # The first visible combat Unit is enough warning to establish a defense.
    # Waiting for it to enter the immediate danger radius leaves a small
    # economy too little time to accumulate the cheapest defender's price.
    if (hostile_units or defense_caution) and combat_units == 0:
        return (UnitType.VANGUARD, UnitType.RANGER)

    if workers < MIN_ECONOMY_WORKERS:
        return (UnitType.WORKER,)

    target_guards = 1 + max(0, (workers - MIN_ECONOMY_WORKERS) // WORKERS_PER_GUARD)
    if hostile_units:
        target_guards += 1
    if combat_units < target_guards:
        preferred = UnitType.VANGUARD if vanguards <= rangers else UnitType.RANGER
        alternate = (
            UnitType.RANGER if preferred is UnitType.VANGUARD else UnitType.VANGUARD
        )
        return (preferred, alternate)
    if workers < MAX_WORKER_POPULATION:
        return (UnitType.WORKER,)
    if turn.state.population < BASE_PRICE_POPULATION_LIMIT:
        preferred = UnitType.VANGUARD if vanguards <= rangers else UnitType.RANGER
        alternate = (
            UnitType.RANGER if preferred is UnitType.VANGUARD else UnitType.VANGUARD
        )
        return (preferred, alternate)
    return ()


def defender_reserve_cost(
    turn,
    enemies: tuple[CoreView | UnitView, ...],
    *,
    defense_caution: bool = False,
) -> int:
    """Return the dynamic defender budget that nonessential upkeep must keep."""

    if not combat_enemies(enemies) and not defense_caution:
        return 0
    return min(
        unit_cost(UnitType.VANGUARD, turn.state.population),
        unit_cost(UnitType.RANGER, turn.state.population),
    )


def defender_reserve_target(turn) -> tuple[UnitType, int]:
    """Return the cheapest dynamically priced combat Unit and its price."""

    candidates = (
        (UnitType.VANGUARD, unit_cost(UnitType.VANGUARD, turn.state.population)),
        (UnitType.RANGER, unit_cost(UnitType.RANGER, turn.state.population)),
    )
    return min(candidates, key=lambda candidate: (candidate[1], candidate[0].value))


def summarize_defense_decision(
    turn,
    enemies: tuple[CoreView | UnitView, ...],
    core_budget: int,
    defense_caution: bool = False,
) -> str | None:
    """Describe the active defender reserve and the Core decision it produced."""

    reserve = defender_reserve_cost(
        turn,
        enemies,
        defense_caution=defense_caution,
    )
    if reserve == 0 or turn.core is None:
        return None
    target, price = defender_reserve_target(turn)
    cell_open = spawn_cell_open(turn)
    action = turn.plan.core_action
    spawned_type = getattr(action, "unit_type", None)
    action_name = (
        type(action).__name__.removesuffix("Action").lower()
        if action is not None
        else "none"
    )
    if spawned_type is not None:
        decision = f"spawn:{spawned_type.value}"
    elif action_name == "heal" and turn.core.hp <= CRITICAL_CORE_HP:
        decision = "heal:critical"
    elif action_name == "repairshield":
        decision = "repair_shield"
    else:
        decision = action_name
    spawn_blockers: list[str] = []
    if not core_is_stationary(turn):
        spawn_blockers.append("core_moving")
    if core_budget < price:
        spawn_blockers.append(f"funds={price - core_budget}")
    if not cell_open:
        spawn_blockers.append("spawn_cell")
    return (
        f"defense[reserve:{reserve},reserve_target:{target.value}@{price},"
        f"bank:{core_budget},cell:{'open' if cell_open else 'blocked'},"
        f"spawn_blockers:{'+'.join(spawn_blockers) if spawn_blockers else '-'},"
        f"decision:{decision}]"
    )


def economic_centroid(points: list[Position]) -> Position | None:
    if not points:
        return None
    return (
        sum(point[0] for point in points) // len(points),
        sum(point[1] for point in points) // len(points),
    )


def start_economic_core_move(
    turn,
    enemies: tuple[CoreView | UnitView, ...],
    budget: TickBudget,
    obstacles: frozenset[Position],
    resources: set[Position],
    reservations: MovementReservations,
    activity_points: list[Position],
    active_economic_workers: int,
    defense_caution: bool,
    memory: ScoutMemory,
    intents: Counter,
) -> bool:
    """Start one safe Core step toward cargo first, then sustained activity."""

    def hold(reason: str) -> bool:
        """Record why migration was skipped so the Tick log can explain a Wait."""

        memory.last_migration_hold = reason
        return False

    core = turn.core
    if core is None or not core_is_stationary(turn):
        return hold("core_not_stationary")
    cargo_workers = sorted(
        (worker for worker in turn.workers if worker.cargo > 0),
        key=lambda worker: (manhattan(worker.position, core.position), str(worker.id)),
    )
    bootstrap_delivery = bool(cargo_workers) and budget.resources < unit_cost(
        UnitType.WORKER,
        turn.state.population,
    )
    # A barren chunk cannot be fixed by escorting carriers around inside it, so
    # the escape outranks both the cargo intercept and the centroid drift.
    escaping = core_should_escape_desert(turn, memory)
    # A lost expedition must not strand the survivors merely because its death
    # lowered population below the normal economic-migration threshold.
    if (
        len(turn.workers) < CORE_RELOCATION_MIN_WORKERS
        and not bootstrap_delivery
        and not escaping
    ):
        return hold("fleet_too_small")
    if defense_caution:
        return hold("recent_core_damage")
    if core.hp < CORE_HP_FLOOR or core.shield < CORE_SHIELD_FLOOR:
        return hold("core_repairing")
    if budget.projected_deposits:
        return hold("deposit_in_flight")
    nearest_cargo = (
        manhattan(cargo_workers[0].position, core.position) if cargo_workers else None
    )
    if nearest_cargo is not None and nearest_cargo <= (
        CORE_MIGRATION_TICKS + CORE_MIGRATION_INCOMING_MARGIN
    ):
        return hold("cargo_already_arriving")
    # Only Units that can actually shoot or charge should pin the Core down.  An
    # enemy Worker or the enemy Core standing nearby used to freeze migration
    # permanently while the telemetry reported no threat at all.
    if any(
        manhattan(core.position, enemy.position) <= CORE_RELOCATION_ENEMY_BUFFER
        for enemy in combat_enemies(enemies)
    ):
        return hold("combat_enemy_near_core")

    def release_intercept() -> None:
        if memory.core_intercept_worker_id is not None:
            memory.core_intercept_worker_id = None
            memory.dirty = True

    # When the bank cannot even fund a Worker, a loaded Worker is guaranteed
    # value.  Intercept the nearest one instead of averaging several distant
    # routes that may cancel each other around the Core.
    if escaping:
        release_intercept()
        goal = (0, 0)
        intents["desert_escaping"] += 1
        intents["density_relocating"] += 1
        log.info(
            "tick=%d core at %s starving in quota-%d chunk; walking toward origin",
            turn.tick,
            core.position,
            chunk_resource_quota(core.position),
        )
    elif bootstrap_delivery:
        worker_by_id = {str(worker.id): worker for worker in cargo_workers}
        locked_worker = worker_by_id.get(memory.core_intercept_worker_id or "")
        if locked_worker is None:
            locked_worker = cargo_workers[0]
            memory.core_intercept_worker_id = str(locked_worker.id)
            memory.dirty = True
        goal = locked_worker.position
    else:
        release_intercept()
        current_quota = chunk_resource_quota(core.position)
        economic_goal = economic_centroid(activity_points)
        if (
            active_economic_workers >= CORE_RELOCATION_MIN_ACTIVE_WORKERS
            and economic_goal is not None
            and chunk_resource_quota(economic_goal) >= current_quota
        ):
            goal = economic_goal
        elif current_quota < CORE_PREFERRED_RESOURCE_QUOTA:
            # Long-lived economic drift otherwise strands the Core in outer
            # chunks whose refill quota is only a fraction of the central area.
            goal = (0, 0)
            intents["density_relocating"] += 1
        else:
            return hold("no_better_chunk")
    if goal is None or manhattan(core.position, goal) < CORE_RELOCATION_MIN_DISTANCE:
        return hold("goal_too_close")

    projected_positions = {
        projected_unit_position(unit, turn.plan) for unit in turn.units
    }
    forbidden = (
        set(obstacles)
        | set(resources)
        | {enemy.position for enemy in enemies}
        | set(reservations.destinations)
    )
    step = step_toward(core.position, goal, forbidden | projected_positions)
    if step is None or step in forbidden or step in projected_positions:
        return hold("no_safe_step")
    core.start_move(direction_between(core.position, step))
    memory.last_migration_hold = None
    intents["relocating"] += 1
    if bootstrap_delivery:
        intents["cargo_intercepting"] += 1
    return True


def plan_core(
    turn,
    enemies: tuple[CoreView | UnitView, ...],
    budget: TickBudget,
    obstacles: frozenset[Position],
    resources: set[Position],
    reservations: MovementReservations,
    activity_points: list[Position],
    active_economic_workers: int,
    defense_caution: bool,
    memory: ScoutMemory,
    intents: Counter,
    settled_capacity: int | None = None,
    healed_units: set[UUID] | None = None,
) -> None:
    core = turn.core
    if core is None:
        return
    if not core_is_stationary(turn):
        destination = core.view.destination
        current_risk = projected_core_damage(core.position, enemies, obstacles)
        destination_risk = (
            projected_core_damage(destination, enemies, obstacles)
            if destination is not None
            else current_risk
        )
        destination_invalid = destination is not None and (
            destination in obstacles
            or destination in resources
            or any(enemy.position == destination for enemy in enemies)
            or destination_risk > current_risk
        )
        required_ticks = core.view.move_required_ticks or CORE_MIGRATION_TICKS
        remaining_ticks = max(0, required_ticks - (core.view.move_progress or 0))
        reaches_safer_cell_now = (
            remaining_ticks <= 1 and destination_risk < current_risk
        )
        exposed_too_long = current_risk > 0 and not reaches_safer_cell_now
        recent_attack_requires_recovery = defense_caution and remaining_ticks > 1
        if destination_invalid or exposed_too_long or recent_attack_requires_recovery:
            core.cancel_move()
            intents["relocation_canceling"] += 1
        else:
            core.wait()
            intents["relocating"] += 1
        return

    population = turn.state.population
    capacity = min(
        turn.resource_capacity,
        settled_capacity if settled_capacity is not None else turn.resource_capacity,
    )
    danger = core_in_danger(
        core.position,
        enemies,
        obstacles,
        DEFENDER_SPAWN_DISTANCE,
    )
    incoming_damage = projected_core_damage(core.position, enemies, obstacles)
    incoming_hp_damage = max(0, incoming_damage - core.shield)

    if core.hp <= CRITICAL_CORE_HP and budget.resources > 0:
        core.heal()
        intents["core_healing"] += 1
        return

    spawn_order = desired_spawn_order(turn, enemies, defense_caution)
    if danger:
        immediate_threats = core_threatening_enemies(core.position, enemies, obstacles)
        adjacent_vanguard = any(
            enemy.unit_type is UnitType.VANGUARD for enemy in immediate_threats
        )
        firing_ranger = any(
            enemy.unit_type is UnitType.RANGER for enemy in immediate_threats
        )
        preferred = (
            UnitType.VANGUARD
            if adjacent_vanguard
            else UnitType.RANGER
            if firing_ranger
            else spawn_order[0]
            if spawn_order and spawn_order[0] is not UnitType.WORKER
            else UnitType.RANGER
        )
        alternate = (
            UnitType.RANGER if preferred is UnitType.VANGUARD else UnitType.VANGUARD
        )
        spawn_order = (preferred, alternate)
    defense_reserve = defender_reserve_cost(
        turn,
        enemies,
        defense_caution=defense_caution,
    )

    def try_spawn() -> bool:
        if not spawn_order or not spawn_cell_open(turn):
            return False
        for unit_type in spawn_order:
            price = unit_cost(unit_type, population)
            if (
                unit_type is UnitType.WORKER
                and budget.resources < price + defense_reserve
            ):
                continue
            first_defender_can_spend = (
                unit_type is not UnitType.WORKER
                and len(turn.vanguards) + len(turn.rangers) == 0
                and core.hp == CORE_HP_FLOOR
                and core.shield >= CORE_SHIELD_FLOOR
                and budget.resources >= price
            )
            emergency_defender_can_spend = (
                danger
                and unit_type is not UnitType.WORKER
                and budget.resources >= price
            )
            reserved_defender_can_spend = (
                defense_reserve > 0
                and unit_type is not UnitType.WORKER
                and budget.resources >= price
            )
            if (
                affordable(price, budget, capacity)
                or first_defender_can_spend
                or emergency_defender_can_spend
                or reserved_defender_can_spend
            ):
                core.spawn(unit_type)
                budget.resources -= price
                intents["producing"] += 1
                return True
        return False

    if danger and try_spawn():
        return

    # HEAL resolves after combat and may be queued at full HP. Once a defender
    # is affordable, producing it removes the damage source and outranks
    # repeatedly buying back one HP. If the spawn cell is temporarily blocked,
    # only the balance above the defender reserve may fund this recovery.
    projected_heal_cost = CORE_HP_FLOOR - core.hp + incoming_hp_damage
    if (
        0 < incoming_hp_damage < core.hp
        and budget.resources >= defense_reserve + projected_heal_cost
    ):
        core.heal()
        intents["core_healing"] += 1
        intents["projected_damage"] += incoming_damage
        return

    if (
        danger
        and core.shield <= CORE_SHIELD_EMERGENCY_FLOOR
        and budget.resources > defense_reserve
    ):
        core.repair_shield()
        intents["shield_repairing"] += 1
        return

    missing_core_hp = CORE_HP_FLOOR - core.hp
    can_heal_without_spending_reserve = budget.resources > 0 and (
        defense_reserve == 0
        or budget.resources >= defense_reserve + missing_core_hp
    )
    if core.hp < CORE_HP_FLOOR and can_heal_without_spending_reserve:
        core.heal()
        intents["core_healing"] += 1
        return
    if core.shield < CORE_SHIELD_FLOOR and budget.resources > defense_reserve:
        core.repair_shield()
        intents["shield_repairing"] += 1
        return
    if not danger and try_spawn():
        return
    if spawn_order:
        next_cost = min(unit_cost(unit_type, population) for unit_type in spawn_order)
        if budget.resources >= max(0, next_cost - CORE_MIGRATION_PRODUCTION_GAP):
            core.wait()
            intents["saving"] += 1
            return
    if (
        not spawn_order
        and core.shield < core_shield_target(turn)
        and budget.resources > max(CORE_HEAL_RESERVE, defense_reserve)
    ):
        core.repair_shield()
        intents["shield_repairing"] += 1
        return
    if healed_units:
        memory.last_migration_hold = "unit_healing"
        core.wait()
        return
    if start_economic_core_move(
        turn,
        enemies,
        budget,
        obstacles,
        resources,
        reservations,
        activity_points,
        active_economic_workers,
        defense_caution,
        memory,
        intents,
    ):
        return
    core.wait()


def beacon_carrier_id(beacon: ChampionBeacon | None) -> UUID | None:
    if beacon is None or beacon.status is not BeaconStatus.CARRIED:
        return None
    return beacon.carrier_id


def choose_beacon_claimer(turn, beacon: ChampionBeacon | None, owned_ids: set[UUID]):
    """Return the owned Unit that should hold or fetch the Champion Beacon.

    Combat Units are preferred as carriers; a cargo-free Worker is used only
    when no soldier is available. Returns None when the Beacon is out of reach
    or already held by an enemy, which combat Units answer by hunting instead.
    """

    if beacon is None:
        return None
    carrier = beacon_carrier_id(beacon)
    if carrier is not None:
        if carrier in owned_ids:
            return next((unit for unit in turn.units if unit.id == carrier), None)
        return None
    if len(turn.units) < BEACON_MIN_UNITS:
        return None
    combat_units = tuple((*turn.vanguards, *turn.rangers))
    preserve_only_guard = len(combat_units) <= 1

    def rank(unit) -> tuple:
        soldier = unit.unit_type is not UnitType.WORKER
        return (
            not soldier,
            getattr(unit, "cargo", 0) > 0,
            manhattan(unit.position, beacon.position),
            str(unit.id),
        )

    candidates = [
        unit
        for unit in turn.units
        if manhattan(unit.position, beacon.position) <= BEACON_CLAIM_DISTANCE
        and not (preserve_only_guard and unit.unit_type is not UnitType.WORKER)
        and not (
            unit.unit_type is UnitType.WORKER
            and unit.position in turn.resource_cells
        )
        and not (
            unit.unit_type is UnitType.WORKER and getattr(unit, "cargo", 0) > 0
        )
    ]
    return min(candidates, key=rank, default=None)


def plan_beacon_unit(
    unit,
    beacon: ChampionBeacon,
    core_position: Position,
    hold_position: Position,
    blocked: set[Position],
    reservations: MovementReservations | None = None,
) -> None:
    """Fetch the Champion Beacon, then hold it next to the Core."""

    if beacon_carrier_id(beacon) == unit.id:
        if unit.position == core_position and hold_position != core_position:
            move_or_wait(unit, hold_position, blocked, reservations)
            return
        if manhattan(unit.position, core_position) <= BEACON_HOLD_DISTANCE:
            unit.wait()
            return
        move_or_wait(unit, core_position, blocked, reservations)
        return
    if unit.position == beacon.position:
        unit.pickup_beacon()
        return
    move_or_wait(unit, beacon.position, blocked, reservations)


def plan_worker(
    worker,
    turn,
    core_position: Position,
    return_position: Position,
    core_receptive: bool,
    blocked: set[Position],
    reservations: MovementReservations,
    resource_target: Position | None,
    visible_resources: frozenset[Position],
    danger: bool,
    budget: TickBudget,
    memory: ScoutMemory,
    scout_goal: Position | None = None,
    hard_blocked: set[Position] | None = None,
    retreat: bool = False,
    guard_core: bool = False,
    preserve_spawn_cell: bool = False,
) -> None:
    position = worker.position
    worker_id = str(worker.id)

    if worker.cargo > 0:
        if preserve_spawn_cell and position == core_position:
            staging = beacon_hold_waypoint(
                core_position,
                blocked | reservations.destinations,
            )
            move_or_wait(worker, staging, blocked, reservations)
            return
        if core_receptive and position == core_position:
            if budget.space > 0:
                amount = min(worker.cargo, budget.space)
                worker.deposit()
                budget.space -= amount
                budget.resources += amount
                budget.projected_deposits += amount
            else:
                worker.wait()
            return
        if position == return_position:
            worker.wait()
            return
        if return_position in blocked and manhattan(position, return_position) == 1:
            worker.wait()
            return
        move_or_escape(
            worker,
            return_position,
            blocked,
            hard_blocked if hard_blocked is not None else blocked,
            reservations,
        )
        return

    if resource_target == position:
        if position in visible_resources:
            worker.harvest()
            return
        if cell_visible_to_friendly(turn, position):
            memory.forget_resource(position)
            log.info("tick=%d forgetting visible stale resource=%s", turn.tick, position)
            resource_target = None

    retreat_stop = 1 if danger else 0
    if retreat and manhattan(position, core_position) > retreat_stop:
        # Reaching the Core cell is what lets plan_unit_heals repair this Worker;
        # stopping alongside it is why a damaged Worker was never healed.  Under
        # threat, retreat_stop keeps that cell clear instead: a Unit projected
        # onto it makes spawn_cell_open() false and would block the emergency
        # defender, and plan_unit_heals skips Workers during danger anyway.
        move_or_escape(
            worker,
            core_position,
            blocked,
            hard_blocked if hard_blocked is not None else blocked,
            reservations,
        )
        return

    if guard_core:
        if manhattan(position, core_position) > 1:
            move_or_wait(worker, core_position, blocked, reservations)
        else:
            worker.wait()
        return

    if resource_target is not None:
        move_or_wait(worker, resource_target, blocked, reservations)
        return

    # Nothing visible to harvest: prefer an unobserved absolute grid cell.  The
    # bounded ray cursor is only a deterministic escape hatch when a frontier
    # target is blocked, and a progress counter for old state.
    legacy_target = scout_target(worker_id, core_position, memory)
    if position == legacy_target:
        advance_scout(worker_id, memory)
        legacy_target = scout_target(worker_id, core_position, memory)
    target = scout_goal or legacy_target
    for _ in range(len(SCOUT_OFFSETS)):
        if manhattan(position, target) <= SCOUT_ARRIVAL_DISTANCE:
            if scout_goal is not None:
                memory.scout_targets.pop(worker_id, None)
                memory.dirty = True
                advance_scout(worker_id, memory)
                scout_goal = None
                target = scout_target(worker_id, core_position, memory)
                continue
            advance_scout(worker_id, memory)
            target = scout_target(worker_id, core_position, memory)
            continue
        if target in blocked:
            if scout_goal is not None:
                memory.scout_targets.pop(worker_id, None)
                memory.dirty = True
                scout_goal = None
            advance_scout(worker_id, memory)
            target = scout_target(worker_id, core_position, memory)
            continue
        if move_or_wait(worker, target, blocked, reservations):
            return
        if scout_goal is not None:
            memory.scout_targets.pop(worker_id, None)
            memory.dirty = True
            advance_scout(worker_id, memory)
            scout_goal = None
            target = scout_target(worker_id, core_position, memory)
            continue
        advance_scout(worker_id, memory)
        target = scout_target(worker_id, core_position, memory)

    # Every waypoint this Tick was unreachable.  The cursor churn above is
    # deliberate — eight steps walk one whole ray, so the next Tick starts on a
    # different ray — but rotating rays cannot help a Worker that is boxed in,
    # and it used to end here in a bare wait() that repeated forever.
    if memory.is_looping(worker_id):
        # Same precedent as a stuck carrier: a Worker that has held one cell for
        # WORKER_STALL_TICKS stops respecting threat arcs rather than idling
        # forever.  One point of damage costs less than a scout that never scouts.
        escape = hard_blocked if hard_blocked is not None else blocked
        if move_or_escape(worker, core_position, escape, escape, reservations):
            log.info(
                "tick=%d worker %s unsticking after %d idle Ticks at %s",
                turn.tick,
                worker_id[:8],
                memory.position_stalls.get(worker_id, 0),
                position,
            )
            return
    worker.wait()


def unit_heal_reserve(
    turn,
    danger: bool,
    defense_caution: bool = False,
) -> int:
    reserve = max(CORE_HEAL_RESERVE, CORE_HP_FLOOR - turn.core.hp)
    reserve = max(
        reserve,
        defender_reserve_cost(
            turn,
            turn.visible_enemies,
            defense_caution=danger or defense_caution,
        ),
    )
    return reserve


def plan_unit_heals(
    turn,
    budget: TickBudget,
    danger: bool,
    intents: Counter,
    defense_caution: bool = False,
) -> set[UUID]:
    """Queue affordable post-combat heals after projected deposits are known."""

    healed: set[UUID] = set()
    if not core_is_stationary(turn):
        return healed
    reserve = unit_heal_reserve(turn, danger, defense_caution)
    for unit in sorted(turn.units, key=lambda candidate: str(candidate.id)):
        if unit.id in turn.plan.unit_actions:
            continue
        if getattr(unit, "cargo", 0) > 0 or unit.position != turn.core.position:
            continue
        if danger and unit.unit_type is UnitType.WORKER:
            continue
        missing = max_hp(unit) - unit.hp
        maximum_cost = max_hp(unit) - 1
        if missing <= 0 or budget.resources < maximum_cost + reserve:
            continue
        unit.heal()
        healed.add(unit.id)
        budget.resources -= maximum_cost
        intents["healing"] += 1
    return healed


def stable_unit_phase(unit_id: UUID, count: int) -> int:
    return sum(str(unit_id).encode()) % count


def guard_waypoint(
    unit,
    core_position: Position,
    obstacles: frozenset[Position],
) -> Position:
    phase = stable_unit_phase(unit.id, len(GUARD_OFFSETS))
    for offset in range(len(GUARD_OFFSETS)):
        dx, dy = GUARD_OFFSETS[(phase + offset) % len(GUARD_OFFSETS)]
        target = (core_position[0] + dx, core_position[1] + dy)
        if target not in obstacles:
            return target
    return core_position


def patrol_waypoint(
    unit,
    core_position: Position,
    tick: int,
    obstacles: frozenset[Position],
) -> Position:
    phase = stable_unit_phase(unit.id, len(PATROL_OFFSETS))
    phase += tick // PATROL_ROTATION_TICKS
    radius = (
        RANGER_PATROL_RADIUS
        if unit.unit_type is UnitType.RANGER
        else VANGUARD_PATROL_RADIUS
    )
    for offset in range(len(PATROL_OFFSETS)):
        dx, dy = PATROL_OFFSETS[(phase + offset) % len(PATROL_OFFSETS)]
        target = (core_position[0] + dx * radius, core_position[1] + dy * radius)
        if target not in obstacles:
            return target
    return core_position


def beacon_hold_waypoint(
    core_position: Position,
    blocked: set[Position],
) -> Position:
    for direction in DIRECTION_ORDER:
        dx, dy = DIRECTION_DELTAS[direction]
        target = (core_position[0] + dx, core_position[1] + dy)
        if target not in blocked:
            return target
    return core_position


def threat_order(
    unit,
    enemies: tuple[CoreView | UnitView, ...],
    carrier: UUID | None,
    core_position: Position,
    obstacles: frozenset[Position],
) -> list:
    """Rank immediate Core threats before carriers and general targets."""

    immediate_threats = {
        enemy.id for enemy in core_threatening_enemies(core_position, enemies, obstacles)
    }

    return sorted(
        enemies,
        key=lambda enemy: (
            enemy.id not in immediate_threats,
            not (
                getattr(enemy, "unit_type", None)
                in {UnitType.VANGUARD, UnitType.RANGER}
                and manhattan(core_position, enemy.position)
                <= CORE_DEFENSE_ALERT_DISTANCE
            ),
            enemy.id != carrier,
            getattr(enemy, "unit_type", None) not in {UnitType.VANGUARD, UnitType.RANGER},
            manhattan(unit.position, enemy.position),
            str(enemy.id),
        ),
    )


def target_durability(enemy: CoreView | UnitView) -> int:
    return enemy.hp + getattr(enemy, "shield", 0)


def needs_planned_damage(enemy: CoreView | UnitView, planned_damage: Counter) -> bool:
    return planned_damage[enemy.id] < target_durability(enemy)


def plan_vanguard(
    vanguard,
    enemies: tuple[CoreView | UnitView, ...],
    blocked: set[Position],
    carrier: UUID | None,
    reservations: MovementReservations,
    idle_target: Position,
    core_position: Position,
    obstacles: frozenset[Position],
    planned_damage: Counter,
) -> None:
    ordered = threat_order(vanguard, enemies, carrier, core_position, obstacles)
    if not ordered:
        move_or_wait(vanguard, idle_target, blocked, reservations)
        return
    unfinished = [
        enemy for enemy in ordered if needs_planned_damage(enemy, planned_damage)
    ]
    candidates = unfinished or ordered
    adjacent = [
        enemy
        for enemy in candidates
        if manhattan(vanguard.position, enemy.position) == 1
    ]
    nearest = adjacent[0] if adjacent else candidates[0]
    dx = nearest.position[0] - vanguard.position[0]
    dy = nearest.position[1] - vanguard.position[1]
    if abs(dx) + abs(dy) == 1:
        vanguard.sweep(direction_between(vanguard.position, nearest.position))
        for enemy in enemies:
            if enemy.position == nearest.position:
                planned_damage[enemy.id] += 1
        return
    move_or_wait(vanguard, nearest.position, blocked, reservations)


def plan_ranger(
    ranger,
    enemies: tuple[CoreView | UnitView, ...],
    obstacles: frozenset[Position],
    blocked: set[Position],
    carrier: UUID | None,
    reservations: MovementReservations,
    idle_target: Position,
    core_position: Position,
    planned_damage: Counter,
) -> None:
    ordered = threat_order(ranger, enemies, carrier, core_position, obstacles)
    unfinished = [
        enemy for enemy in ordered if needs_planned_damage(enemy, planned_damage)
    ]
    for enemy in unfinished:
        if is_legal_shot(ranger.position, enemy.position, obstacles):
            ranger.shoot(enemy)
            planned_damage[enemy.id] += 1
            return
    if unfinished:
        move_or_wait(ranger, unfinished[0].position, blocked, reservations)
        return
    for enemy in ordered:
        if is_legal_shot(ranger.position, enemy.position, obstacles):
            ranger.shoot(enemy)
            planned_damage[enemy.id] += 1
            return
    if not ordered:
        move_or_wait(ranger, idle_target, blocked, reservations)
        return
    move_or_wait(ranger, ordered[0].position, blocked, reservations)


def max_hp(unit) -> int:
    if unit.unit_type is UnitType.VANGUARD:
        return 4
    return 2


def worker_should_retreat(worker, hostile_units) -> bool:
    """Decide whether this Worker should be walking home instead of working.

    Damage used to change a Worker's behaviour in no way at all: a Worker on its
    last hit point kept scouting exactly as if it were fresh, and because it
    never came home the heal path — which only fires on the Core cell — had
    never once run.  Replacing a Worker costs five; healing one costs one.
    """

    if worker.hp < max_hp(worker):
        return True
    return any(
        manhattan(worker.position, enemy.position) <= WORKER_FLEE_DISTANCE
        for enemy in hostile_units
    )


def choose_danger_worker_guard(
    workers,
    core_position: Position,
    worker_blocked: set[Position],
    excluded_ids: set[UUID],
    hostile_units: tuple[UnitView, ...],
) -> UUID | None:
    """Hold one safe Worker only when it already blocks a Vanguard approach."""

    approach_cells: set[Position] = set()
    for enemy in hostile_units:
        if enemy.unit_type is not UnitType.VANGUARD:
            continue
        current_distance = manhattan(core_position, enemy.position)
        for direction in DIRECTION_ORDER:
            dx, dy = DIRECTION_DELTAS[direction]
            cell = (core_position[0] + dx, core_position[1] + dy)
            if manhattan(cell, enemy.position) < current_distance:
                approach_cells.add(cell)

    candidates = (
        worker
        for worker in workers
        if worker.id not in excluded_ids
        and worker.cargo == 0
        and worker.hp == max_hp(worker)
        and worker.position not in worker_blocked
        and worker.position in approach_cells
    )
    guard = min(
        candidates,
        key=lambda worker: (manhattan(worker.position, core_position), str(worker.id)),
        default=None,
    )
    return guard.id if guard is not None else None


def decide(turn, memory: ScoutMemory | None = None) -> None:
    """Queue one complete plan for the Turn using the balanced policy."""

    if memory is None:
        memory = ScoutMemory()
    memory.last_intents = Counter()
    memory.last_migration_hold = None
    memory.last_defense_status = None
    observe(turn, memory)
    if turn.state.status is not PlayerStatus.ACTIVE or turn.core is None:
        for unit in turn.units:
            unit.wait()
        return

    enemies = turn.visible_enemies
    enemy_cells = {enemy.position for enemy in enemies}
    core_position = turn.core.position
    remember_obstacles(turn, memory)
    obstacles = frozenset(memory.known_obstacles)
    blocked = blocked_cells(turn.units, core_position, obstacles, enemy_cells)
    worker_blocked = blocked | remember_threat_cells(turn, memory, obstacles)
    remember_resources(turn, memory)
    mark_scout_coverage(turn, memory)
    for worker in turn.workers:
        memory.record_position(str(worker.id), worker.position)
    visible_resources = turn.resource_cells
    resources = available_resources(turn, memory)

    hostile_units = combat_enemies(enemies)
    danger = core_in_danger(
        core_position,
        enemies,
        obstacles,
        DANGER_DISTANCE,
    )
    if core_in_danger(
        core_position,
        enemies,
        obstacles,
        CORE_DEFENSE_ALERT_DISTANCE,
    ):
        previous_caution = memory.core_threat_until_tick
        memory.core_threat_until_tick = max(
            previous_caution,
            turn.tick + CORE_THREAT_CAUTION_TICKS,
        )
        memory.dirty |= memory.core_threat_until_tick != previous_caution
    defense_caution = turn.tick <= memory.core_threat_until_tick
    worker_travel_blocked = worker_blocked | ({core_position} if danger else set())
    worker_hard_blocked = blocked | ({core_position} if danger else set())

    settled_capacity = projected_post_combat_capacity(turn, enemies, obstacles)
    capacity_at_risk = settled_capacity < turn.resource_capacity
    settled_resources = (
        min(turn.resources, settled_capacity) if capacity_at_risk else turn.resources
    )
    budget = TickBudget(
        resources=settled_resources,
        space=(
            min(
                turn.resource_space,
                max(0, settled_capacity - settled_resources),
            )
            if capacity_at_risk
            else turn.resource_space
        ),
    )
    defense_reserve = defender_reserve_cost(
        turn,
        enemies,
        defense_caution=defense_caution,
    )
    reservations = MovementReservations(destinations=set(memory.contested))
    if memory.contested:
        log.info("tick=%d backoff contested=%s", turn.tick, sorted(memory.contested))
    intents: Counter = Counter()

    beacon = turn.beacon
    owned_ids = {unit.id for unit in turn.units}
    claimer = choose_beacon_claimer(turn, beacon, owned_ids)
    claimer_id = claimer.id if claimer is not None else None

    hostile_carrier = beacon_carrier_id(beacon)
    if hostile_carrier in owned_ids:
        hostile_carrier = None

    core_receptive = core_is_stationary(turn)
    return_position = core_position
    if not core_receptive and turn.core.view.destination is not None:
        return_position = turn.core.view.destination
    rendezvous_keeper = None
    if not core_receptive:
        destination_occupants = sorted(
            (unit for unit in turn.units if unit.position == return_position),
            key=lambda unit: str(unit.id),
        )
        if destination_occupants:
            rendezvous_keeper = destination_occupants[0].id
            reservations.destinations.add(return_position)

    cargo_workers = sorted(
        (worker for worker in turn.workers if worker.cargo > 0),
        key=lambda worker: (manhattan(worker.position, return_position), str(worker.id)),
    )
    for worker in cargo_workers:
        preserve_spawn_cell = (
            danger
            and defense_reserve > 0
            and budget.resources >= defense_reserve
        )
        cargo_travel_blocked = (
            worker_travel_blocked if preserve_spawn_cell else worker_blocked
        )
        cargo_hard_blocked = (
            worker_hard_blocked if preserve_spawn_cell else blocked
        )
        if preserve_spawn_cell and worker.position == core_position:
            intents["evacuating"] += 1
        elif core_receptive and worker.position == core_position and budget.space > 0:
            intents["depositing"] += 1
        else:
            intents["returning"] += 1
        if (
            not core_receptive
            and worker.position == return_position
            and worker.id != rendezvous_keeper
        ):
            staging = beacon_hold_waypoint(
                return_position,
                cargo_hard_blocked | reservations.destinations | {core_position},
            )
            move_or_wait(worker, staging, cargo_hard_blocked, reservations)
            continue
        avoid = cargo_travel_blocked
        if memory.is_looping(str(worker.id)):
            # Shuffling between the same two cells delivers nothing, so a stuck
            # carrier stops respecting threat arcs and takes the direct route.
            # One point of damage costs less than an expedition that never ends.
            avoid = cargo_hard_blocked
            intents["loop_breaking"] += 1
            log.info(
                "tick=%d worker %s loops with cargo at %s; ignoring threat arcs",
                turn.tick,
                str(worker.id)[:8],
                worker.position,
            )
        plan_worker(
            worker,
            turn,
            core_position,
            return_position,
            core_receptive,
            avoid,
            reservations,
            None,
            visible_resources,
            danger,
            budget,
            memory,
            hard_blocked=cargo_hard_blocked,
            retreat=worker_should_retreat(worker, hostile_units),
            preserve_spawn_cell=preserve_spawn_cell,
        )

    # Deposits resolve before healing and the Core action. Planning them first
    # makes their same-Tick budget available without assuming a move can deposit.
    healed = plan_unit_heals(
        turn,
        budget,
        danger,
        intents,
        defense_caution,
    )

    danger_guard_id = (
        choose_danger_worker_guard(
            turn.workers,
            core_position,
            worker_travel_blocked,
            healed | ({claimer_id} if claimer_id is not None else set()),
            hostile_units,
        )
        if danger
        else None
    )

    mining_workers = [
        worker
        for worker in turn.workers
        if worker.id not in healed
        and worker.id != claimer_id
        and worker.cargo == 0
        and worker.id != danger_guard_id
    ]
    refresh_resource_progress(
        mining_workers,
        resources,
        worker_travel_blocked,
        turn.tick,
        memory,
    )
    max_remote_workers = remote_worker_limit(
        turn.workers,
        resources,
        return_position,
        worker_travel_blocked,
        hostile_units,
        defense_caution,
    )
    remote_commitments = sum(
        manhattan(worker.position, return_position) > RESOURCE_LOCAL_RETURN_DISTANCE
        for worker in cargo_workers
    )
    trip_budget = resource_round_trip_budget(turn, memory)
    memory.last_trip_budget = trip_budget

    def assign(max_total_cost: int, max_remote_workers: int) -> dict:
        return assign_resource_targets(
            mining_workers,
            resources,
            worker_travel_blocked,
            memory.resource_assignments,
            depot=return_position,
            max_total_cost=max_total_cost,
            tick=turn.tick,
            last_seen=memory.resource_last_seen,
            cooldowns=memory.resource_cooldowns,
            remote_distance=RESOURCE_LOCAL_RETURN_DISTANCE,
            max_remote_workers=max_remote_workers,
        )

    remote_slots = max(0, max_remote_workers - remote_commitments)
    resource_targets = assign(trip_budget, remote_slots)
    if (
        not resource_targets
        and mining_workers
        and resources
        and trip_budget < RESOURCE_TRIP_COST_HEALTHY
    ):
        # The drought budget tightens trips to protect a weak economy, but when it
        # refuses every node the fleet walks past food it already found.  Raise the
        # cost cap only; the fleet-wide expedition limit still applies.
        resource_targets = assign(RESOURCE_TRIP_COST_HEALTHY, remote_slots)
        if resource_targets:
            memory.last_trip_budget = RESOURCE_TRIP_COST_HEALTHY
            intents["stretching_trip"] += 1
            log.info(
                "tick=%d stretching one trip past budget %d to %s",
                turn.tick,
                trip_budget,
                sorted(resource_targets.values()),
            )
    persisted_assignments = {
        str(worker_id): target for worker_id, target in resource_targets.items()
    }
    if persisted_assignments != memory.resource_assignments:
        memory.resource_assignments = persisted_assignments
        memory.dirty = True
    for worker_id, target in persisted_assignments.items():
        progress = memory.resource_progress.get(worker_id)
        if progress is None or progress.target != target:
            worker = next(
                worker for worker in mining_workers if str(worker.id) == worker_id
            )
            cost = bounded_route_cost(worker.position, target, worker_travel_blocked)
            memory.resource_progress[worker_id] = ResourceProgress(
                target,
                cost if cost is not None else PATH_COST_UNREACHABLE,
            )
        if worker_id in memory.scout_targets:
            memory.scout_targets.pop(worker_id, None)
            memory.dirty = True
        memory.recalling_workers.discard(worker_id)
    for worker_id in set(memory.resource_progress) - set(persisted_assignments):
        memory.resource_progress.pop(worker_id, None)
    if resource_targets:
        assignments = ",".join(
            f"{str(worker_id)[:8]}->{target}@"
            f"{bounded_route_cost(turn.unit(worker_id).position, target, worker_travel_blocked)}"
            for worker_id, target in sorted(resource_targets.items(), key=lambda item: str(item[0]))
        )
        log.debug(
            "tick=%d resources visible=%d known=%d assignments[%s]",
            turn.tick,
            len(visible_resources),
            len(memory.known_resources),
            assignments,
        )

    eligible_workers = [
        worker
        for worker in turn.workers
        if worker.id not in healed
        and worker.id != claimer_id
        and worker.cargo == 0
    ]
    miners = sorted(
        (
            worker
            for worker in eligible_workers
            if worker.cargo == 0 and worker.id in resource_targets
        ),
        key=lambda worker: (
            manhattan(worker.position, resource_targets[worker.id]),
            str(worker.id),
        ),
    )
    scouts = sorted(
        (
            worker
            for worker in eligible_workers
            if worker.cargo == 0 and worker.id not in resource_targets
        ),
        key=lambda worker: str(worker.id),
    )

    if claimer is not None and claimer.id not in healed:
        intents["guarding"] += 1
        plan_beacon_unit(
            claimer,
            beacon,
            core_position,
            beacon_hold_waypoint(core_position, blocked | reservations.destinations),
            worker_travel_blocked if claimer.unit_type is UnitType.WORKER else blocked,
            reservations,
        )
    for worker in miners:
        intents["mining"] += 1
        plan_worker(
            worker,
            turn,
            core_position,
            return_position,
            core_receptive,
            worker_travel_blocked,
            reservations,
            resource_targets[worker.id],
            visible_resources,
            danger,
            budget,
            memory,
            hard_blocked=worker_hard_blocked,
            retreat=worker_should_retreat(worker, hostile_units),
        )

    combat_units = sorted(
        (
            unit
            for unit in (*turn.vanguards, *turn.rangers)
            if unit.id not in healed and unit.id != claimer_id
        ),
        key=lambda unit: str(unit.id),
    )
    planned_damage: Counter = Counter()
    immediate_threat_ids = {
        enemy.id
        for enemy in core_threatening_enemies(core_position, enemies, obstacles)
    }
    defense_enemies = tuple(
        enemy
        for enemy in hostile_units
        if enemy.id in immediate_threat_ids
        or manhattan(core_position, enemy.position) <= CORE_DEFENSE_ALERT_DISTANCE
    )
    for index, unit in enumerate(combat_units):
        if defense_enemies:
            engagement_targets = defense_enemies
            intents["engaging"] += 1
            idle_target = guard_waypoint(unit, core_position, obstacles)
        elif enemies and index > 0:
            engagement_targets = enemies
            intents["engaging"] += 1
            idle_target = guard_waypoint(unit, core_position, obstacles)
        elif index == 0:
            engagement_targets = ()
            intents["guarding"] += 1
            idle_target = guard_waypoint(unit, core_position, obstacles)
        else:
            engagement_targets = ()
            intents["scouting"] += 1
            idle_target = patrol_waypoint(unit, core_position, turn.tick, obstacles)
        if unit.unit_type is UnitType.VANGUARD:
            plan_vanguard(
                unit,
                engagement_targets,
                blocked,
                hostile_carrier,
                reservations,
                idle_target,
                core_position,
                obstacles,
                planned_damage,
            )
        else:
            plan_ranger(
                unit,
                engagement_targets,
                obstacles,
                blocked,
                hostile_carrier,
                reservations,
                idle_target,
                core_position,
                planned_damage,
            )

    claimed_scout_cells: set[tuple[int, int]] = set()
    scout_radius = scout_disc_radius(core_position, memory)
    scout_tether = scout_radius + SCOUT_TETHER_MARGIN
    if scout_radius > SCOUT_MAX_DISTANCE:
        intents["searching_wide"] += 1
    for worker in scouts:
        worker_id = str(worker.id)
        distance_from_core = manhattan(worker.position, core_position)
        if distance_from_core > scout_tether:
            memory.recalling_workers.add(worker_id)
        if (
            worker_id in memory.recalling_workers
            and distance_from_core <= SCOUT_SAFE_RETURN_DISTANCE
        ):
            memory.recalling_workers.discard(worker_id)
        if worker_id in memory.recalling_workers:
            memory.scout_targets.pop(worker_id, None)
            memory.dirty = True
            intents["recalling"] += 1
            move_or_wait(worker, core_position, worker_travel_blocked, reservations)
            continue
        retreating = worker_should_retreat(worker, hostile_units)
        if worker.id == danger_guard_id:
            intents["guarding"] += 1
        elif retreating:
            intents["evading"] += 1
        else:
            intents["scouting"] += 1
        plan_worker(
            worker,
            turn,
            core_position,
            return_position,
            core_receptive,
            worker_travel_blocked,
            reservations,
            None,
            visible_resources,
            danger,
            budget,
            memory,
            scout_coverage_target(
                worker,
                core_position,
                memory,
                claimed_scout_cells,
                turn.tick,
                worker_travel_blocked,
                scout_radius,
            ),
            hard_blocked=worker_hard_blocked,
            retreat=retreating,
            guard_core=worker.id == danger_guard_id,
        )

    # Unit movement is now fully planned, so rerun the capacity preview against
    # the actual destinations before choosing the Core action. Deposits happen
    # before combat and Unit heals happen after it; model that order when
    # rebuilding the remaining Core budget after a newly exposed casualty.
    settled_capacity = projected_post_combat_capacity(turn, enemies, obstacles)
    if settled_capacity < turn.resource_capacity:
        planned_unit_heal_cost = sum(
            max_hp(unit) - 1
            for unit in turn.units
            if unit.id in healed
        )
        pre_heal_resources = min(
            turn.resources + budget.projected_deposits,
            settled_capacity,
        )
        budget.resources = max(0, pre_heal_resources - planned_unit_heal_cost)
        budget.space = max(0, settled_capacity - pre_heal_resources)

    activity_points: list[Position] = []
    for worker in cargo_workers:
        activity_points.extend((worker.position, worker.position))
    for worker in miners:
        activity_points.append(worker.position)
        activity_points.extend((resource_targets[worker.id], resource_targets[worker.id]))
    for cell in visible_resources:
        activity_points.extend((cell, cell))
    active_economic_workers = len(cargo_workers) + len(miners)

    core_budget = budget.resources
    plan_core(
        turn,
        enemies,
        budget,
        obstacles,
        resources,
        reservations,
        activity_points,
        active_economic_workers,
        defense_caution,
        memory,
        intents,
        settled_capacity,
        healed,
    )
    memory.last_defense_status = summarize_defense_decision(
        turn,
        enemies,
        core_budget,
        defense_caution,
    )

    if reservations.by_unit:
        moves = ",".join(
            f"{str(unit_id)[:8]}->{destination}"
            for unit_id, destination in sorted(
                reservations.by_unit.items(), key=lambda item: str(item[0])
            )
        )
        log.debug("tick=%d reserved moves[%s]", turn.tick, moves)

    memory.last_intents = intents
    memory.last_move_destinations = {
        str(unit_id): destination
        for unit_id, destination in reservations.by_unit.items()
    }


def load_api_key(env_path: str | None) -> str:
    if env_path is not None and os.path.exists(env_path):
        for line in Path(env_path).read_text().splitlines():
            stripped = line.strip().removeprefix("export ").lstrip()
            if stripped.startswith("ARENA_HERO_API_KEY="):
                return stripped.split("=", 1)[1].strip().strip("\"'")
    key = os.environ.get("ARENA_HERO_API_KEY")
    if key:
        return key
    raise SystemExit("ARENA_HERO_API_KEY not found in environment or .env file")


def summarize(plan) -> str:
    counts = Counter(
        type(action).__name__.removesuffix("Action") for action in plan.unit_actions.values()
    )
    core = type(plan.core_action).__name__.removesuffix("Action") if plan.core_action else None
    parts = ",".join(f"{name}:{count}" for name, count in sorted(counts.items()))
    return f"units[{parts or 'none'}] core[{core or 'none'}]"


def summarize_fleet(turn) -> str:
    """Expose the living composition hidden by the aggregate population."""

    return (
        f"fleet[w:{len(turn.workers)},v:{len(turn.vanguards)},"
        f"r:{len(turn.rangers)}]"
    )


def summarize_events(counts: Counter) -> str:
    if not counts:
        return "events[none]"
    parts = ",".join(f"{name}:{count}" for name, count in sorted(counts.items()))
    return f"events[{parts}]"


def summarize_intents(counts: Counter) -> str:
    primary = (
        "returning",
        "depositing",
        "mining",
        "scouting",
        "recalling",
        "guarding",
    )
    parts = [f"{name}:{counts[name]}" for name in primary]
    parts.extend(
        f"{name}:{count}"
        for name, count in sorted(counts.items())
        if name not in primary and count
    )
    return f"intent[{','.join(parts)}]"


def summarize_resource_flow(flow: Counter) -> str:
    """Compactly expose actual resource amounts from the previous Tick."""

    names = ("harvest", "deposit", "capture", "dropped", "overflow", "other")
    return "flow[" + ",".join(f"{name}:{flow[name]}" for name in names) + "]"


def summarize_economy(memory: ScoutMemory, tick: int) -> str:
    """Expose the rolling feedback that controls remote trip length."""

    totals = memory.economic_totals(tick)
    return (
        f"econ[{ECONOMY_FLOW_WINDOW}:h{totals['harvest']},"
        f"i{totals['income']},loss{totals['lost']},"
        f"n{totals['samples']},trip{memory.last_trip_budget}]"
    )


def nearest_deposit_eta(
    turn,
    obstacles: frozenset[Position] | set[Position] = frozenset(),
) -> int | None:
    """Report the soonest deposit in route Ticks rather than straight-line cells."""

    if turn.core is None:
        return None
    cargo_workers = [worker for worker in turn.workers if worker.cargo > 0]
    if not cargo_workers:
        return None
    target = turn.core.position
    migration_delay = 0
    if turn.core.view.state is CoreState.MOVING:
        target = turn.core.view.destination or target
        migration_delay = max(
            0,
            (turn.core.view.move_required_ticks or CORE_MIGRATION_TICKS)
            - (turn.core.view.move_progress or 0),
        )
    # Manhattan distance is an admissible lower bound on route cost, so walking
    # Workers nearest-first lets the search stop as soon as the next lower bound
    # cannot beat the best route already measured.  A Worker with a wall between
    # it and the Core no longer reports an impossibly early arrival.
    ordered = sorted(
        cargo_workers,
        key=lambda worker: (manhattan(worker.position, target), str(worker.id)),
    )
    best: int | None = None
    for worker in ordered:
        if best is not None and manhattan(worker.position, target) >= best:
            break
        cost = bounded_route_cost(worker.position, target, obstacles)
        if cost is not None and (best is None or cost < best):
            best = cost
    if best is None:
        return None
    return max(best, migration_delay)


def summarize_core_state(
    turn,
    obstacles: frozenset[Position] | set[Position],
) -> str:
    """Expose the minimum Core telemetry needed to audit defense decisions."""

    if turn.core is None:
        return "base[none]"
    hostile_units = combat_enemies(turn.visible_enemies)
    nearest_enemy = min(
        (manhattan(turn.core.position, enemy.position) for enemy in hostile_units),
        default=None,
    )
    # Harmless enemies (Workers, the enemy Core) still influence migration and
    # pathing, so report them separately instead of leaving `enemy:-` implying an
    # empty neighbourhood.
    combat_ids = {id(enemy) for enemy in hostile_units}
    nearest_other = min(
        (
            manhattan(turn.core.position, enemy.position)
            for enemy in turn.visible_enemies
            if id(enemy) not in combat_ids
        ),
        default=None,
    )
    incoming = projected_core_damage(
        turn.core.position,
        turn.visible_enemies,
        frozenset(obstacles),
    )
    state = turn.core.view.state.value
    if turn.core.view.state is CoreState.MOVING:
        state = (
            f"MOVING:{turn.core.view.move_progress or 0}/"
            f"{turn.core.view.move_required_ticks or CORE_MIGRATION_TICKS}"
        )
    return (
        f"base[pos:{turn.core.position},hp:{turn.core.hp},shield:{turn.core.shield},"
        f"state:{state},enemy:{nearest_enemy if nearest_enemy is not None else '-'},"
        f"other:{nearest_other if nearest_other is not None else '-'},"
        f"incoming:{incoming}]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="/root/arena-hero/.env", help="Path to the .env file")
    parser.add_argument(
        "--state",
        default="/root/arena-hero/scout_state.json",
        help="Path to the file that carries scout progress across restarts",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--log-file",
        default=None,
        help="Write tactic logs through a rotating file handler",
    )
    args = parser.parse_args()

    logging_kwargs = {
        "level": logging.DEBUG if args.verbose else logging.INFO,
        "format": "%(asctime)s %(levelname)s %(message)s",
    }
    if args.log_file:
        logging_kwargs["handlers"] = [
            RotatingFileHandler(
                args.log_file,
                maxBytes=TACTIC_LOG_MAX_BYTES,
                backupCount=TACTIC_LOG_BACKUPS,
                encoding="utf-8",
            )
        ]
    logging.basicConfig(**logging_kwargs)
    # The SDK's HTTP client logs one INFO line per accepted command.  Keep
    # tactic.log focused on decisions and resolution events.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    api_key = load_api_key(args.env)
    memory = ScoutMemory(path=Path(args.state))
    memory.load()

    with ArenaHeroClient(api_key=api_key) as game:
        log.info("connected, waiting for Turns")
        for turn in game.turns():
            planning_started = time.perf_counter()
            try:
                decide(turn, memory)
            except Exception:
                log.exception("planning failed on tick %d, submitting empty plan", turn.tick)
                try:
                    turn.clear()
                    turn.submit()
                except Exception:
                    log.exception("empty fallback plan failed on tick %d", turn.tick)
                continue

            planning_ms = (time.perf_counter() - planning_started) * 1000
            try:
                turn.submit()
            except APIError as exc:
                if exc.error in {"TICK_MISMATCH", "COMMAND_WINDOW_CLOSED"}:
                    log.warning(
                        "tick=%d submission expired error=%s; waiting for fresh state",
                        turn.tick,
                        exc.error,
                    )
                else:
                    log.exception(
                        "tick=%d submission rejected error=%s",
                        turn.tick,
                        exc.error,
                    )
                continue
            except Exception:
                log.exception("submission failed on tick %d", turn.tick)
                continue

            memory.save(turn.tick)
            deposit_eta = nearest_deposit_eta(turn, memory.known_obstacles)
            log.info(
                "tick=%d status=%s resources=%d/%d pop=%d %s %s %s %s %s %s map[r:%d,o:%d] "
                "deposit_eta=%s hold=%s plan_ms=%.1f %s %s",
                turn.tick,
                turn.state.status.value,
                turn.resources,
                turn.resource_capacity,
                turn.state.population,
                summarize_fleet(turn),
                summarize_intents(memory.last_intents),
                summarize_resource_flow(memory.last_resource_flow),
                summarize_economy(memory, turn.tick),
                summarize_core_state(turn, memory.known_obstacles),
                memory.last_defense_status or "defense[-]",
                len(memory.known_resources),
                len(memory.known_obstacles),
                deposit_eta if deposit_eta is not None else "-",
                memory.last_migration_hold or "-",
                planning_ms,
                summarize(turn.plan),
                summarize_events(memory.last_events),
            )


if __name__ == "__main__":
    main()
