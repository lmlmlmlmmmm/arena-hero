"""Cross-Tick tactic state and persistence helpers.

This module intentionally has no dependency on the live planner.  The planner
imports and re-exports the state classes from :mod:`tactic` for compatibility.
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

from arena_hero import Position

from tactic_config import (
    ECONOMY_FLOW_WINDOW,
    ECONOMY_HISTORY_LIMIT,
    OBSTACLE_MEMORY_KEEP_CELLS,
    OBSTACLE_MEMORY_MAX_CELLS,
    RESOURCE_ACTIVE_ASSIGNMENT_TTL,
    RESOURCE_MEMORY_TTL,
    RESOURCE_TRIP_COST_NORMAL,
    SCOUT_ABSOLUTE_GRID_SCHEMA,
    SCOUT_COVERAGE_MAX_CELLS,
    SCOUT_COVERAGE_TTL,
    SCOUT_HISTORY_LIMIT,
    SCOUT_LOOP_WINDOW,
    SCOUT_SAVE_INTERVAL,
    SCOUT_SECTOR_COUNT,
    SCOUT_STATE_SCHEMA,
    WORKER_STALL_TICKS,
)

log = logging.getLogger("arena-hero-tactic")


def manhattan(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _coverage_max_cells() -> int:
    """Read the facade alias so legacy monkeypatches keep working.

    Older tests and integrations patch ``tactic.SCOUT_COVERAGE_MAX_CELLS``
    directly.  The value normally comes from :mod:`tactic_config`, but a
    lazy lookup preserves that supported module-level customization after
    the state class moved here.
    """

    planner = sys.modules.get("tactic")
    if planner is not None:
        value = getattr(planner, "SCOUT_COVERAGE_MAX_CELLS", None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return SCOUT_COVERAGE_MAX_CELLS


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
    # A Worker keeps its assigned cardinal scouting sector across target
    # rotations and tactic restarts.  New Workers fill the least-populated
    # sector; existing Workers are never reshuffled.
    scout_sector_slots: dict[str, int] = field(default_factory=dict)
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
    core_identity: str | None = None
    core_migration_goal: Position | None = None
    core_migration_goal_kind: str | None = None
    core_last_move_delta: Position | None = None
    core_last_move_tick: int = -1
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
        raw_core_identity = raw.get("core_identity")
        self.core_identity = (
            raw_core_identity
            if isinstance(raw_core_identity, str) and raw_core_identity
            else None
        )
        self.core_migration_goal = parse_position(raw.get("core_migration_goal"))
        raw_goal_kind = raw.get("core_migration_goal_kind")
        self.core_migration_goal_kind = (
            raw_goal_kind
            if raw_goal_kind in {"activity", "density"}
            else None
        )
        last_delta = parse_position(raw.get("core_last_move_delta"))
        self.core_last_move_delta = (
            last_delta
            if last_delta is not None
            and abs(last_delta[0]) + abs(last_delta[1]) == 1
            else None
        )
        raw_last_move_tick = raw.get("core_last_move_tick", -1)
        self.core_last_move_tick = (
            raw_last_move_tick
            if isinstance(raw_last_move_tick, int)
            and not isinstance(raw_last_move_tick, bool)
            else -1
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
            self.scout_sector_slots = {}
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
            self.scout_sector_slots = {
                worker_id: sector
                for worker_id, sector in parse_int_map(
                    raw.get("scout_sector_slots", {})
                ).items()
                if 0 <= sector < SCOUT_SECTOR_COUNT
            }
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
                        "scout_sector_slots": self.scout_sector_slots,
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
                        "core_identity": self.core_identity,
                        "core_migration_goal": (
                            list(self.core_migration_goal)
                            if self.core_migration_goal is not None
                            else None
                        ),
                        "core_migration_goal_kind": self.core_migration_goal_kind,
                        "core_last_move_delta": (
                            list(self.core_last_move_delta)
                            if self.core_last_move_delta is not None
                            else None
                        ),
                        "core_last_move_tick": self.core_last_move_tick,
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

    def sync_core_identity(self, core_id: str) -> None:
        """Discard migration intent when the server replaces the Core."""

        if self.core_identity == core_id:
            return
        replaced = self.core_identity is not None
        self.core_identity = core_id
        if not replaced:
            self.dirty = True
            return
        self.core_intercept_worker_id = None
        self.core_migration_goal = None
        self.core_migration_goal_kind = None
        self.core_last_move_delta = None
        self.core_last_move_tick = -1
        self.dirty = True

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
            self.scout_sector_slots,
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
        overflow = len(self.scout_seen) - _coverage_max_cells()
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
