"""Combat threat evaluation and unit tactics for Arena Hero."""

from __future__ import annotations

from collections import Counter

from arena_hero import CoreView, Position, UnitType, UnitView

from tactic_config import CORE_DEFENSE_ALERT_DISTANCE
from tactic_pathing import direction_between, manhattan, move_or_wait


def is_legal_shot(
    origin: Position,
    cell: Position,
    obstacles: frozenset[Position],
) -> bool:
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
    return not any(
        (origin[0] + i * step_x, origin[1] + i * step_y) in obstacles
        for i in range(1, distance)
    )


def combat_enemies(
    enemies: tuple[CoreView | UnitView, ...],
) -> tuple[UnitView, ...]:
    """Return visible enemies that can damage a Core or Unit."""

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


def threat_order(
    unit,
    enemies: tuple[CoreView | UnitView, ...],
    carrier,
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
            getattr(enemy, "unit_type", None)
            not in {UnitType.VANGUARD, UnitType.RANGER},
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
    carrier,
    reservations,
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
    if manhattan(vanguard.position, nearest.position) == 1:
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
    carrier,
    reservations,
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


__all__ = [
    "combat_enemies",
    "core_in_danger",
    "core_threatening_enemies",
    "is_legal_shot",
    "needs_planned_damage",
    "plan_ranger",
    "plan_vanguard",
    "projected_core_damage",
    "target_durability",
    "threat_order",
]
