"""Balanced Arena Hero tactic: economy, defense, Beacon control, and upkeep.

Reads the API key from the ARENA_HERO_API_KEY environment variable or from the
file given by --env, connects with the official SDK, and submits one plan per
Tick. Terrain knowledge and scouting progress survive restarts through the
--state file.
"""

from __future__ import annotations

import argparse
import logging
from logging.handlers import RotatingFileHandler
import os
import sys
import time
from collections import Counter
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

# Policy constants live in a dependency-leaf module. Re-exporting them here
# keeps the historical ``tactic.CONSTANT`` API intact.
from tactic_config import *  # noqa: F401,F403
from tactic_pathing import (
    bounded_route_cost,
    direction_between,
    manhattan,
    move_or_escape,
    move_or_wait,
    nearest_target,
    step_toward,
)
from tactic_combat import (
    combat_enemies,
    core_in_danger,
    core_threatening_enemies,
    is_legal_shot,
    needs_planned_damage,
    plan_ranger,
    plan_vanguard,
    projected_core_damage,
    target_durability,
    threat_order,
)
from tactic_state import (
    MovementReservations,
    ResourceProgress,
    ScoutMemory,
    TickBudget,
)

log = logging.getLogger("arena-hero-tactic")


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


def chunk_ring(position: Position) -> int:
    """Return the documented density ring for a world position."""

    chunk_x, chunk_y = position[0] // 32, position[1] // 32

    def axis(value: int) -> int:
        return value if value >= 0 else -value - 1

    return axis(chunk_x) + axis(chunk_y)


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


def scout_sector_for_worker(
    worker_id: str,
    memory: ScoutMemory | None = None,
) -> int:
    """Return a Worker sector, allocating a durable least-filled slot.

    Without a memory object this keeps the original deterministic UUID-based
    fallback for callers that only need a stable sector in isolation.  The
    live planner passes ``ScoutMemory`` so assignments are balanced once and
    then retained across target rotations and restarts.
    """

    if memory is not None:
        existing = memory.scout_sector_slots.get(worker_id)
        if existing is not None and 0 <= existing < SCOUT_SECTOR_COUNT:
            return existing
        counts = Counter(
            sector
            for sector in memory.scout_sector_slots.values()
            if 0 <= sector < SCOUT_SECTOR_COUNT
        )
        sector = min(
            range(SCOUT_SECTOR_COUNT),
            key=lambda candidate: (counts[candidate], candidate),
        )
        memory.scout_sector_slots[worker_id] = sector
        memory.dirty = True
        return sector

    try:
        return UUID(worker_id).bytes[-1] % SCOUT_SECTOR_COUNT
    except (ValueError, AttributeError):
        # Test doubles and old persisted state may carry a non-UUID key. Keep
        # the assignment deterministic instead of making scouting fail closed.
        return sum(worker_id.encode()) % SCOUT_SECTOR_COUNT


def ensure_scout_sector_slots(
    worker_ids: set[str] | list[str] | tuple[str, ...],
    memory: ScoutMemory,
) -> None:
    """Allocate balanced sectors for every currently living Worker."""

    living = set(worker_ids)
    for worker_id in sorted(living):
        scout_sector_for_worker(worker_id, memory)


def remote_scout_role_count(worker_count: int) -> int:
    """Return the bounded remote share for the current Worker fleet."""

    return (
        max(0, worker_count) * SCOUT_REMOTE_ROLE_NUMERATOR
        // SCOUT_REMOTE_ROLE_DENOMINATOR
    )


def ensure_scout_roles(
    worker_ids: set[str] | list[str] | tuple[str, ...],
    memory: ScoutMemory,
) -> None:
    """Keep a deterministic local/remote split across rotations and restarts."""

    living = set(worker_ids)
    ensure_scout_sector_slots(living, memory)
    changed = False
    for worker_id in set(memory.scout_roles) - living:
        del memory.scout_roles[worker_id]
        changed = True
    for worker_id in sorted(living):
        if memory.scout_roles.get(worker_id) not in {"local", "remote"}:
            memory.scout_roles[worker_id] = "local"
            changed = True

    target_remote = remote_scout_role_count(len(living))
    remote = {
        worker_id
        for worker_id in living
        if memory.scout_roles.get(worker_id) == "remote"
    }
    while len(remote) < target_remote:
        sector_counts = Counter(memory.scout_sector_slots[item] for item in remote)
        candidate = min(
            living - remote,
            key=lambda worker_id: (
                sector_counts[memory.scout_sector_slots[worker_id]],
                memory.scout_sector_slots[worker_id],
                worker_id,
            ),
        )
        memory.scout_roles[candidate] = "remote"
        remote.add(candidate)
        changed = True
    while len(remote) > target_remote:
        sector_counts = Counter(memory.scout_sector_slots[item] for item in remote)
        candidate = max(
            remote,
            key=lambda worker_id: (
                sector_counts[memory.scout_sector_slots[worker_id]],
                memory.scout_sector_slots[worker_id],
                worker_id,
            ),
        )
        memory.scout_roles[candidate] = "local"
        remote.remove(candidate)
        changed = True
    if changed:
        memory.dirty = True


def scout_radius_for_worker(worker_id: str, memory: ScoutMemory) -> int:
    """Return the fixed radius belonging to one Worker's durable scout role."""

    return (
        SCOUT_FAR_DISTANCE
        if memory.scout_roles.get(worker_id) == "remote"
        else SCOUT_MAX_DISTANCE
    )


def scout_sector(cell: Position, core_position: Position) -> int:
    """Return the cardinal sector containing a cell around the Core.

    Sectors are ordered East, South, West, North. Ties are assigned to the
    horizontal axis so every grid cell has exactly one owner sector.
    """

    dx = cell[0] - core_position[0]
    dy = cell[1] - core_position[1]
    if abs(dx) >= abs(dy):
        return 0 if dx >= 0 else 2
    return 1 if dy >= 0 else 3


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
    min_distance: int = 0,
) -> Position | None:
    """Select a stable absolute grid target within the current scout disc."""

    worker_id = str(worker.id)
    preferred_sector = scout_sector_for_worker(worker_id, memory)
    existing = memory.scout_targets.get(worker_id)
    if existing is not None and existing not in claimed:
        target = scout_grid_center(existing)
        core_distance = manhattan(core_position, target)
        if (
            core_distance > max_distance
            or core_distance < min_distance
            or target in blocked
        ):
            log.info(
                "tick=%d worker %s dropping out-of-range scout target=%s",
                tick,
                worker_id[:8],
                target,
            )
            memory.scout_targets.pop(worker_id, None)
            memory.dirty = True
        elif scout_sector(target, core_position) != preferred_sector:
            log.info(
                "tick=%d worker %s moving scout target to sector=%d",
                tick,
                worker_id[:8],
                preferred_sector,
            )
            memory.scout_targets.pop(worker_id, None)
            if memory.is_looping(worker_id):
                memory.scout_seen[existing] = tick
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
    candidate_cells = [
        cell
        for cell in scout_grid_disc(core_position, max_distance)
        if manhattan(core_position, scout_grid_center(cell)) >= min_distance
    ]
    if not candidate_cells and min_distance:
        candidate_cells = scout_grid_disc(core_position, max_distance)

    candidates: list[tuple[int, int, int, int, int, int, int, tuple[int, int]]] = []
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
        sector_penalty = int(
            scout_sector(target, core_position) != preferred_sector
        )
        candidates.append(
            (
                sector_penalty,
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
    _, _, _, _, _, _, _, cell = min(candidates)
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
    combat: Counter = Counter()
    for event in turn.events:
        counts[event.event_type] += 1
        values = event.values if isinstance(event.values, dict) else {}
        if event.event_type == "SWEEP_RESOLVED":
            combat["sweeps"] += 1
            targets_hit = values.get("targets_hit", 0)
            if isinstance(targets_hit, int) and not isinstance(targets_hit, bool):
                combat["sweep_targets"] += max(0, targets_hit)
        elif event.event_type == "SHOT_HIT":
            combat["ranger_hits"] += 1
            damage = values.get("damage", 0)
            if isinstance(damage, int) and not isinstance(damage, bool):
                combat["ranger_damage"] += max(0, damage)
        elif event.event_type == "SHOT_MISSED":
            combat["ranger_misses"] += 1
        elif event.event_type == "DESTRUCTION_PARTICIPATION":
            combat["destructions"] += 1
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
        if event.event_type in {
            "CORE_MOVE_START_FAILED",
            "CORE_MOVE_FAILED",
            "CORE_MOVE_CANCELLED",
        }:
            clear_core_move(memory)
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
    memory.last_combat_results = combat
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
        return (preferred,)

    # The first visible combat Unit is enough warning to establish a defense.
    # Waiting for it to enter the immediate danger radius leaves a small
    # economy too little time to accumulate the cheapest defender's price.
    if (hostile_units or defense_caution) and combat_units == 0:
        return (UnitType.VANGUARD,)

    if workers < MIN_ECONOMY_WORKERS:
        return (UnitType.WORKER,)

    target_guards = 1 + max(0, (workers - MIN_ECONOMY_WORKERS) // WORKERS_PER_GUARD)
    if hostile_units:
        target_guards += 1
    if combat_units < target_guards:
        preferred = UnitType.VANGUARD if vanguards <= rangers else UnitType.RANGER
        return (preferred,)
    if workers < MAX_WORKER_POPULATION:
        return (UnitType.WORKER,)
    # Keep enough ranged coverage even if an older policy already filled the
    # base-price population band entirely with Vanguards.  There is no Unit
    # maintenance charge, so correcting that composition above 20 is cheaper
    # than waiting for a casualty or Core respawn.
    if rangers < MIN_RANGER_POPULATION:
        return (UnitType.RANGER,)
    if turn.state.population < BASE_PRICE_POPULATION_LIMIT:
        preferred = UnitType.VANGUARD if vanguards <= rangers else UnitType.RANGER
        return (preferred,)
    return ()


def effective_spawn_order(
    turn,
    enemies: tuple[CoreView | UnitView, ...],
    defense_caution: bool,
    danger: bool,
    obstacles: frozenset[Position],
) -> tuple[UnitType, ...]:
    """Return the production order after applying immediate-threat overrides."""

    spawn_order = desired_spawn_order(turn, enemies, defense_caution)
    if not danger or turn.core is None:
        return spawn_order
    immediate_threats = core_threatening_enemies(
        turn.core.position,
        enemies,
        obstacles,
    )
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
    return (preferred, alternate)


def defender_reserve_cost(
    turn,
    enemies: tuple[CoreView | UnitView, ...],
    *,
    defense_caution: bool = False,
    spawn_order: tuple[UnitType, ...] = (),
) -> int:
    """Return the dynamic defender budget that nonessential upkeep must keep."""

    if not combat_enemies(enemies) and not defense_caution:
        return 0
    combat_order = tuple(
        unit_type for unit_type in spawn_order if unit_type is not UnitType.WORKER
    )
    unit_types = combat_order or (UnitType.VANGUARD, UnitType.RANGER)
    return min(
        unit_cost(unit_type, turn.state.population) for unit_type in unit_types
    )


def defender_reserve_target(
    turn,
    spawn_order: tuple[UnitType, ...] = (),
) -> tuple[UnitType, int]:
    """Return the cheapest combat Unit the current production policy accepts."""

    combat_order = tuple(
        unit_type for unit_type in spawn_order if unit_type is not UnitType.WORKER
    )
    unit_types = combat_order or (UnitType.VANGUARD, UnitType.RANGER)
    candidates = tuple(
        (unit_type, unit_cost(unit_type, turn.state.population))
        for unit_type in unit_types
    )
    return min(candidates, key=lambda candidate: (candidate[1], candidate[0].value))


def summarize_defense_decision(
    turn,
    enemies: tuple[CoreView | UnitView, ...],
    core_budget: int,
    defense_caution: bool = False,
) -> str | None:
    """Describe the active defender reserve and the Core decision it produced."""

    if turn.core is None:
        return None
    danger = core_in_danger(
        turn.core.position,
        enemies,
        turn.obstacle_cells,
        DEFENDER_SPAWN_DISTANCE,
    )
    spawn_order = effective_spawn_order(
        turn,
        enemies,
        defense_caution,
        danger,
        turn.obstacle_cells,
    )
    reserve = defender_reserve_cost(
        turn,
        enemies,
        defense_caution=defense_caution,
        spawn_order=spawn_order,
    )
    if reserve == 0:
        return None
    target, price = defender_reserve_target(turn, spawn_order)
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
    combat_order = tuple(
        unit_type for unit_type in spawn_order if unit_type is not UnitType.WORKER
    )
    if not core_is_stationary(turn):
        spawn_blockers.append("core_moving")
    if not spawn_order:
        spawn_blockers.append("policy_population_cap")
    elif not combat_order:
        spawn_blockers.append("policy_economy_priority")
    elif core_budget < price:
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


def core_migration_milestone(
    origin: Position,
    destination: Position,
    limit: int = CORE_DENSITY_MILESTONE_DISTANCE,
) -> Position:
    """Return an x-then-y waypoint no more than one chunk from ``origin``."""

    if limit <= 0:
        return origin
    remaining = limit
    x, y = origin
    dx = destination[0] - x
    step_x = min(abs(dx), remaining)
    x += step_x if dx >= 0 else -step_x
    remaining -= step_x
    dy = destination[1] - y
    step_y = min(abs(dy), remaining)
    y += step_y if dy >= 0 else -step_y
    return (x, y)


def clear_core_migration_goal(memory: ScoutMemory) -> None:
    """Release a completed or obsolete multi-step economic destination."""

    if memory.core_migration_goal is None and memory.core_migration_goal_kind is None:
        return
    memory.core_migration_goal = None
    memory.core_migration_goal_kind = None
    memory.dirty = True


def remember_core_migration_goal(
    memory: ScoutMemory,
    goal: Position,
    kind: str,
) -> None:
    """Keep economic movement aimed at one destination across Core steps."""

    if memory.core_migration_goal == goal and memory.core_migration_goal_kind == kind:
        return
    memory.core_migration_goal = goal
    memory.core_migration_goal_kind = kind
    memory.dirty = True


def remember_core_move(memory: ScoutMemory, direction: Direction, tick: int) -> None:
    """Record the latest economic step so short-lived target noise cannot reverse it."""

    delta = DIRECTION_DELTAS[direction]
    if memory.core_last_move_delta == delta and memory.core_last_move_tick == tick:
        return
    memory.core_last_move_delta = delta
    memory.core_last_move_tick = tick
    memory.dirty = True


def clear_core_move(memory: ScoutMemory) -> None:
    """Forget an in-progress direction when migration ends without displacement."""

    if memory.core_last_move_delta is None and memory.core_last_move_tick == -1:
        return
    memory.core_last_move_delta = None
    memory.core_last_move_tick = -1
    memory.dirty = True


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
    if core is None:
        return hold("core_not_stationary")
    # Normalize legacy persisted density goals before any economic early-return
    # gate (cargo arrival, repair, or danger) can leave a multi-hour target in
    # telemetry or resume it on a later Tick.
    if (
        memory.core_migration_goal_kind == "density"
        and memory.core_migration_goal is not None
        and manhattan(core.position, memory.core_migration_goal)
        > CORE_DENSITY_MILESTONE_DISTANCE
    ):
        remember_core_migration_goal(
            memory,
            core_migration_milestone(core.position, memory.core_migration_goal),
            "density",
        )
    if not core_is_stationary(turn):
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
        clear_core_migration_goal(memory)
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
        remembered_goal = memory.core_migration_goal
        remembered_kind = memory.core_migration_goal_kind
        if (
            remembered_kind == "density"
            and remembered_goal is not None
            and manhattan(core.position, remembered_goal)
            > CORE_DENSITY_MILESTONE_DISTANCE
        ):
            # Old persisted state may contain the pre-milestone origin target.
            # Trim it before it can resume a multi-hour locked march.
            remembered_goal = core_migration_milestone(core.position, remembered_goal)
            remember_core_migration_goal(memory, remembered_goal, "density")
        current_quota = chunk_resource_quota(core.position)
        economic_goal = economic_centroid(activity_points)
        remembered_goal_is_valid = (
            remembered_kind == "density"
            and current_quota < CORE_PREFERRED_RESOURCE_QUOTA
        ) or (
            remembered_kind == "activity"
            and active_economic_workers >= CORE_RELOCATION_MIN_ACTIVE_WORKERS
            and economic_goal is not None
            and remembered_goal is not None
            and chunk_resource_quota(remembered_goal) >= current_quota
        )
        if (
            remembered_goal is not None
            and remembered_goal_is_valid
            and manhattan(core.position, remembered_goal)
            >= CORE_RELOCATION_MIN_DISTANCE
        ):
            goal = remembered_goal
            goal_kind = remembered_kind
            intents["migration_goal_locked"] += 1
            if goal_kind == "density":
                intents["density_relocating"] += 1
        else:
            if remembered_goal is not None:
                clear_core_migration_goal(memory)
                reason = (
                    "goal_reached"
                    if manhattan(core.position, remembered_goal)
                    < CORE_RELOCATION_MIN_DISTANCE
                    else "goal_invalidated"
                )
                return hold(reason)
            if (
                active_economic_workers >= CORE_RELOCATION_MIN_ACTIVE_WORKERS
                and economic_goal is not None
                and chunk_resource_quota(economic_goal) >= current_quota
            ):
                goal = economic_goal
                goal_kind = "activity"
            elif current_quota < CORE_PREFERRED_RESOURCE_QUOTA:
                # Long-lived economic drift otherwise strands the Core in outer
                # chunks whose refill quota is only a fraction of the central area.
                goal = core_migration_milestone(core.position, (0, 0))
                goal_kind = "density"
                intents["density_relocating"] += 1
            else:
                return hold("no_better_chunk")
            remember_core_migration_goal(memory, goal, goal_kind)
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
    direction = direction_between(core.position, step)
    delta = DIRECTION_DELTAS[direction]
    last_delta = memory.core_last_move_delta
    reversing = (
        last_delta is not None
        and delta == (-last_delta[0], -last_delta[1])
    )
    if (
        reversing
        and not bootstrap_delivery
        and not escaping
        and turn.tick - memory.core_last_move_tick
        < CORE_MIGRATION_REVERSE_COOLDOWN_TICKS
    ):
        return hold("reverse_cooldown")
    core.start_move(direction)
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
        if core.view.move_direction is not None:
            inferred_start_tick = turn.tick - (core.view.move_progress or 1)
            if inferred_start_tick > memory.core_last_move_tick:
                remember_core_move(
                    memory,
                    core.view.move_direction,
                    inferred_start_tick,
                )
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

    spawn_order = effective_spawn_order(
        turn,
        enemies,
        defense_caution,
        danger,
        obstacles,
    )
    defense_reserve = defender_reserve_cost(
        turn,
        enemies,
        defense_caution=defense_caution,
        spawn_order=spawn_order,
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
    reserve_caution = danger or defense_caution
    spawn_danger = core_in_danger(
        turn.core.position,
        turn.visible_enemies,
        turn.obstacle_cells,
        DEFENDER_SPAWN_DISTANCE,
    )
    spawn_order = effective_spawn_order(
        turn,
        turn.visible_enemies,
        reserve_caution,
        spawn_danger,
        turn.obstacle_cells,
    )
    reserve = max(CORE_HEAL_RESERVE, CORE_HP_FLOOR - turn.core.hp)
    reserve = max(
        reserve,
        defender_reserve_cost(
            turn,
            turn.visible_enemies,
            defense_caution=reserve_caution,
            spawn_order=spawn_order,
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
    memory.sync_core_identity(str(turn.core.id))

    # Allocate once per living Worker. Existing sectors and local/remote roles
    # survive target rotation and restarts; only a changed fleet is rebalanced.
    ensure_scout_roles({str(worker.id) for worker in turn.workers}, memory)

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
    defense_spawn_danger = core_in_danger(
        core_position,
        enemies,
        obstacles,
        DEFENDER_SPAWN_DISTANCE,
    )
    defense_spawn_order = effective_spawn_order(
        turn,
        enemies,
        defense_caution,
        defense_spawn_danger,
        obstacles,
    )
    defense_reserve = defender_reserve_cost(
        turn,
        enemies,
        defense_caution=defense_caution,
        spawn_order=defense_spawn_order,
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
    carrier_enemies = tuple(
        enemy for enemy in enemies if enemy.id == hostile_carrier
    )
    for index, unit in enumerate(combat_units):
        # A Vanguard already adjacent to an enemy must spend this Tick's action
        # on that contact. The old role split assigned the first combat unit to
        # guarding whenever the Core was safe, even if that Vanguard was
        # face-to-face with an enemy outside the Core alert ring.
        adjacent_enemies = (
            tuple(
                enemy
                for enemy in enemies
                if manhattan(unit.position, enemy.position) == 1
            )
            if unit.unit_type is UnitType.VANGUARD
            else ()
        )
        if adjacent_enemies:
            engagement_targets = adjacent_enemies
            intents["engaging"] += 1
            idle_target = guard_waypoint(unit, core_position, obstacles)
        elif defense_enemies:
            engagement_targets = defense_enemies
            intents["engaging"] += 1
            idle_target = guard_waypoint(unit, core_position, obstacles)
        elif carrier_enemies and index < COMBAT_PURSUIT_HUNTERS:
            engagement_targets = carrier_enemies
            intents["engaging"] += 1
            intents["carrier_hunting"] += 1
            idle_target = guard_waypoint(unit, core_position, obstacles)
        elif 0 < index <= COMBAT_PURSUIT_HUNTERS:
            engagement_targets = tuple(
                enemy
                for enemy in enemies
                if manhattan(unit.position, enemy.position)
                <= COMBAT_PURSUIT_DISTANCE
            )
            if engagement_targets:
                intents["engaging"] += 1
                intents["limited_pursuit"] += 1
            else:
                intents["scouting"] += 1
            idle_target = patrol_waypoint(unit, core_position, turn.tick, obstacles)
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
    for worker in scouts:
        worker_id = str(worker.id)
        scout_radius = scout_radius_for_worker(worker_id, memory)
        scout_tether = scout_radius + SCOUT_TETHER_MARGIN
        scout_role = memory.scout_roles.get(worker_id, "local")
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
            intents[f"scouting_{scout_role}"] += 1
            if scout_role == "remote":
                intents["searching_wide"] += 1
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
                SCOUT_MAX_DISTANCE + 1 if scout_role == "remote" else 0,
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


def summarize_combat(results: Counter) -> str:
    """Expose resolved attack effectiveness from the previous Tick."""

    return (
        f"combat[sweeps:{results['sweeps']},"
        f"targets:{results['sweep_targets']},"
        f"shots:{results['ranger_hits']}/{results['ranger_misses']},"
        f"damage:{results['ranger_damage']},"
        f"destructions:{results['destructions']}]"
    )


def summarize_economy(memory: ScoutMemory, tick: int) -> str:
    """Expose the rolling feedback that controls remote trip length."""

    totals = memory.economic_totals(tick)
    return (
        f"econ[{ECONOMY_FLOW_WINDOW}:h{totals['harvest']},"
        f"i{totals['income']},loss{totals['lost']},"
        f"n{totals['samples']},trip{memory.last_trip_budget}]"
    )


def summarize_migration(turn, memory: ScoutMemory) -> str:
    """Expose chunk economics and the currently locked migration milestone."""

    if turn.core is None:
        return "migration[none]"
    goal = memory.core_migration_goal
    distance = manhattan(turn.core.position, goal) if goal is not None else None
    return (
        f"migration[ring:{chunk_ring(turn.core.position)},"
        f"quota:{chunk_resource_quota(turn.core.position)},"
        f"kind:{memory.core_migration_goal_kind or '-'},"
        f"goal:{goal if goal is not None else '-'},"
        f"distance:{distance if distance is not None else '-'}]"
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
                "tick=%d status=%s resources=%d/%d pop=%d %s %s %s %s %s %s %s %s map[r:%d,o:%d] "
                "deposit_eta=%s hold=%s plan_ms=%.1f %s %s",
                turn.tick,
                turn.state.status.value,
                turn.resources,
                turn.resource_capacity,
                turn.state.population,
                summarize_fleet(turn),
                summarize_intents(memory.last_intents),
                summarize_resource_flow(memory.last_resource_flow),
                summarize_combat(memory.last_combat_results),
                summarize_economy(memory, turn.tick),
                summarize_core_state(turn, memory.known_obstacles),
                summarize_migration(turn, memory),
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
