"""Offline tests for the balanced tactic decision logic (no network, no key)."""

from __future__ import annotations

import uuid
from collections import Counter
from collections.abc import Callable

from arena_hero import (
    BeaconStatus,
    ChampionBeacon,
    CoreView,
    CoreState,
    Direction,
    PlayerState,
    PlayerStatus,
    ResolutionEvent,
    TerrainView,
    Turn,
    UnitType,
    UnitView,
)

from tactic import (
    ScoutMemory,
    TickBudget,
    affordable,
    decide,
    is_legal_shot,
    projected_post_combat_capacity,
    scout_target,
)

U = lambda n: uuid.UUID(f"00000000-0000-4000-8000-0000000000{n:02x}")

# Far enough that the default fixture never diverts a Unit to the Beacon.
FAR_BEACON = ChampionBeacon(position=(500, 500))


def make_turn(
    *,
    status: PlayerStatus = PlayerStatus.ACTIVE,
    respawn_at_tick: int | None = None,
    resources: int = 5,
    objects: tuple = (),
    events: tuple = (),
    beacon: ChampionBeacon = None,
    tick: int = 100,
) -> Turn:
    captured: list = []

    def submitter(plan, key=None):
        captured.append((plan, key))
        return None

    state = PlayerState(
        status=status,
        respawn_at_tick=respawn_at_tick,
        resources=resources,
        population=sum(
            1 for obj in objects if isinstance(obj, UnitView) and obj.controlled
        ),
        champion_beacon=beacon or FAR_BEACON,
        objects=objects,
        events=events,
    )
    turn = Turn(tick=tick, state=state, submitter=submitter)
    turn._captured = captured
    return turn


def core_view(
    position=(0, 0),
    hp=5,
    shield=5,
    uid=1,
    state=CoreState.NORMAL,
    move_direction=None,
    move_progress=None,
    move_required_ticks=None,
    destination=None,
):
    values = dict(
        kind="CORE",
        id=U(uid),
        controlled=True,
        owner_username="testplayer",
        position=position,
        hp=hp,
        shield=shield,
        state=state,
    )
    if state is CoreState.MOVING:
        values.update(
            move_direction=move_direction or Direction.RIGHT,
            move_progress=move_progress if move_progress is not None else 1,
            move_required_ticks=(
                move_required_ticks if move_required_ticks is not None else 4
            ),
            destination=destination or (position[0] + 1, position[1]),
        )
    return CoreView(**values)


def worker_view(position, cargo=None, hp=2, uid=2):
    return UnitView(
        kind="UNIT",
        id=U(uid),
        controlled=True,
        position=position,
        hp=hp,
        unit_type=UnitType.WORKER,
        cargo=cargo,
    )


def ranger_view(position, hp=2, uid=3):
    return UnitView(
        kind="UNIT",
        id=U(uid),
        controlled=True,
        position=position,
        hp=hp,
        unit_type=UnitType.RANGER,
    )


def vanguard_view(position, hp=4, uid=4):
    return UnitView(
        kind="UNIT",
        id=U(uid),
        controlled=True,
        position=position,
        hp=hp,
        unit_type=UnitType.VANGUARD,
    )


def enemy_view(
    position,
    hp=2,
    uid=90,
    kind="UNIT",
    unit_type=UnitType.VANGUARD,
):
    if kind == "CORE":
        return CoreView(
            kind="CORE",
            id=U(uid),
            controlled=False,
            owner_username="enemyplayer",
            position=position,
            hp=5,
            shield=5,
            state=CoreState.NORMAL,
        )
    return UnitView(
        kind="UNIT",
        id=U(uid),
        controlled=False,
        position=position,
        hp=hp,
        unit_type=unit_type,
    )


def terrain(kind: str, positions) -> TerrainView:
    return TerrainView(kind=kind, positions=tuple(positions))


def action_type(plan, unit_id):
    action = plan.unit_actions.get(U(unit_id))
    return type(action).__name__ if action is not None else None


def direction_of(plan, unit_id):
    action = plan.unit_actions.get(U(unit_id))
    return action.direction if action is not None else None


def core_action_type(turn):
    plan = turn.plan
    return type(plan.core_action).__name__ if plan.core_action is not None else None


def test_active_no_resources_worker_scouts():
    turn = make_turn(
        resources=10,
        objects=(core_view(), worker_view((3, 0))),
    )
    decide(turn)
    assert action_type(turn.plan, 2) == "MoveAction"


def test_worker_harvests_on_resource_cell():
    turn = make_turn(
        resources=10,
        objects=(core_view(), worker_view((2, 0)), terrain("RESOURCE", [(2, 0)])),
    )
    decide(turn)
    assert action_type(turn.plan, 2) == "HarvestAction"


def test_worker_deposits_at_core():
    turn = make_turn(
        resources=5,
        objects=(core_view(), worker_view((0, 0), cargo=1)),
    )
    decide(turn)
    assert action_type(turn.plan, 2) == "DepositAction"


def test_worker_with_cargo_moves_toward_core():
    turn = make_turn(
        resources=5,
        objects=(core_view(), worker_view((5, 0), cargo=2)),
    )
    decide(turn)
    assert action_type(turn.plan, 2) == "MoveAction"
    assert direction_of(turn.plan, 2) == Direction.LEFT


def test_cargo_worker_remembers_obstacles_instead_of_backtracking():
    memory = ScoutMemory()
    first = make_turn(
        tick=100,
        objects=(
            core_view(),
            worker_view((3, 3), cargo=1),
            terrain("OBSTACLE", [(2, 3), (3, 2)]),
        ),
    )
    decide(first, memory)
    assert direction_of(first.plan, 2) == Direction.RIGHT
    assert memory.known_obstacles == {(2, 3), (3, 2)}

    # The walls are now outside the current Turn's visible terrain. Remembering
    # permanent obstacles keeps the route from immediately stepping left again.
    second = make_turn(
        tick=101,
        objects=(core_view(), worker_view((4, 3), cargo=1)),
    )
    decide(second, memory)
    assert direction_of(second.plan, 2) == Direction.UP


def test_worker_contention_one_harvests():
    turn = make_turn(
        resources=10,
        objects=(
            core_view(),
            worker_view((2, 0), uid=2),
            worker_view((3, 0), uid=3),
            terrain("RESOURCE", [(2, 0)]),
        ),
    )
    decide(turn)
    assert action_type(turn.plan, 2) == "HarvestAction"
    assert action_type(turn.plan, 3) == "MoveAction"


def test_core_spawns_worker_when_affordable():
    turn = make_turn(
        resources=10,
        objects=(core_view(), worker_view((3, 0))),
    )
    decide(turn)
    assert core_action_type(turn) == "SpawnAction"
    assert turn.plan.core_action.unit_type is UnitType.WORKER


def test_safe_roster_waits_for_preferred_ranger_instead_of_cheaper_vanguard():
    workers = tuple(worker_view((20 + uid, 0), uid=uid) for uid in range(2, 16))
    vanguards = tuple(vanguard_view((uid, 2), uid=uid) for uid in range(16, 20))
    turn = make_turn(
        resources=12,
        objects=(core_view(), *workers, *vanguards),
    )
    decide(turn)
    assert core_action_type(turn) == "WaitAction"


def test_safe_roster_spawns_preferred_ranger_after_heal_reserve_is_funded():
    workers = tuple(worker_view((20 + uid, 0), uid=uid) for uid in range(2, 16))
    vanguards = tuple(vanguard_view((uid, 2), uid=uid) for uid in range(16, 20))
    turn = make_turn(
        resources=14,
        objects=(core_view(), *workers, *vanguards),
    )
    decide(turn)
    assert core_action_type(turn) == "SpawnAction"
    assert turn.plan.core_action.unit_type is UnitType.RANGER


def test_legacy_twenty_unit_roster_adds_missing_ranged_coverage():
    workers = tuple(worker_view((20 + uid, 0), uid=uid) for uid in range(2, 16))
    vanguards = tuple(vanguard_view((uid, 2), uid=uid) for uid in range(16, 22))
    turn = make_turn(
        resources=18,
        objects=(core_view(), *workers, *vanguards),
    )
    decide(turn)
    assert core_action_type(turn) == "SpawnAction"
    assert turn.plan.core_action.unit_type is UnitType.RANGER


def test_core_heals_when_damaged():
    turn = make_turn(
        resources=10,
        objects=(core_view(hp=3), worker_view((3, 0))),
    )
    decide(turn)
    assert core_action_type(turn) == "HealAction"


def test_core_repairs_shield():
    turn = make_turn(
        resources=10,
        objects=(core_view(hp=5, shield=2), worker_view((3, 0))),
    )
    decide(turn)
    assert core_action_type(turn) == "RepairShieldAction"


def test_projected_ranger_hp_damage_preserves_defender_reserve():
    turn = make_turn(
        resources=5,
        objects=(
            core_view(shield=0),
            worker_view((4, 0), uid=2),
            enemy_view((3, 0), unit_type=UnitType.RANGER),
        ),
    )
    decide(turn)
    assert core_action_type(turn) == "WaitAction"


def test_projected_ranger_hp_damage_spawns_affordable_defender_before_healing():
    turn = make_turn(
        resources=10,
        objects=(
            core_view(shield=0),
            worker_view((4, 0), uid=2),
            enemy_view((3, 0), unit_type=UnitType.RANGER),
        ),
    )
    decide(turn)
    assert core_action_type(turn) == "SpawnAction"
    assert turn.plan.core_action.unit_type is UnitType.VANGUARD


def test_danger_spawns_before_post_combat_emergency_shield_repair():
    turn = make_turn(
        resources=20,
        objects=(
            core_view(shield=2),
            worker_view((4, 0), uid=2),
            worker_view((4, 1), uid=3),
            enemy_view((2, 0)),
        ),
    )
    decide(turn)
    assert core_action_type(turn) == "SpawnAction"
    assert turn.plan.core_action.unit_type is UnitType.RANGER


def test_danger_saves_for_cheapest_defender_instead_of_repairing_shield():
    turn = make_turn(
        resources=8,
        objects=(
            core_view(shield=2),
            worker_view((4, 0), uid=2),
            enemy_view((2, 0)),
        ),
    )
    decide(turn)
    assert core_action_type(turn) == "WaitAction"


def test_danger_spawns_defender_as_soon_as_reserve_reaches_its_price():
    turn = make_turn(
        resources=10,
        objects=(
            core_view(shield=2),
            worker_view((4, 0), uid=2),
            enemy_view((1, 0)),
        ),
    )
    decide(turn)
    assert core_action_type(turn) == "SpawnAction"
    assert turn.plan.core_action.unit_type is UnitType.VANGUARD


def test_defense_telemetry_reports_the_reserved_defender_shortfall():
    memory = ScoutMemory()
    turn = make_turn(
        resources=8,
        objects=(
            core_view(shield=2),
            worker_view((4, 0), uid=2),
            enemy_view((2, 0)),
        ),
    )
    decide(turn, memory)
    assert memory.last_defense_status is not None
    assert "reserve:10" in memory.last_defense_status
    assert "reserve_target:VANGUARD@10" in memory.last_defense_status
    assert "spawn_blockers:funds=2" in memory.last_defense_status
    assert "decision:wait" in memory.last_defense_status


def test_defense_telemetry_reports_policy_population_cap():
    memory = ScoutMemory()
    workers = tuple(worker_view((20 + uid, 0), uid=uid) for uid in range(2, 16))
    vanguards = tuple(vanguard_view((uid, 2), uid=uid) for uid in range(16, 20))
    rangers = tuple(ranger_view((uid, 3), uid=uid) for uid in range(20, 22))
    turn = make_turn(
        resources=18,
        objects=(
            core_view(),
            *workers,
            *vanguards,
            *rangers,
            enemy_view((9, 0)),
        ),
    )
    decide(turn, memory)
    assert memory.last_defense_status is not None
    assert "spawn_blockers:policy_population_cap" in memory.last_defense_status


def test_defense_telemetry_prices_the_only_policy_approved_defender():
    memory = ScoutMemory()
    workers = tuple(worker_view((20 + uid, 0), uid=uid) for uid in range(2, 16))
    vanguards = tuple(vanguard_view((uid, 2), uid=uid) for uid in range(16, 20))
    turn = make_turn(
        resources=11,
        objects=(
            core_view(),
            *workers,
            *vanguards,
            enemy_view((9, 0)),
        ),
    )
    decide(turn, memory)
    assert memory.last_defense_status is not None
    assert "reserve_target:RANGER@12" in memory.last_defense_status
    assert "spawn_blockers:funds=1" in memory.last_defense_status


def test_defense_telemetry_reports_worker_first_economy_policy():
    memory = ScoutMemory()
    turn = make_turn(
        resources=15,
        objects=(
            core_view(),
            worker_view((5, 0), uid=2),
            worker_view((6, 0), uid=3),
            worker_view((7, 0), uid=4),
            vanguard_view((2, 0), uid=5),
            enemy_view((9, 0)),
        ),
    )
    decide(turn, memory)
    assert memory.last_defense_status is not None
    assert "spawn_blockers:policy_economy_priority" in memory.last_defense_status
    assert "policy_population_cap" not in memory.last_defense_status


def test_safe_ranger_reserve_outranks_nonessential_shield_repair():
    workers = tuple(worker_view((20 + uid, 0), uid=uid) for uid in range(2, 16))
    vanguards = tuple(vanguard_view((uid, 2), uid=uid) for uid in range(16, 20))
    turn = make_turn(
        resources=12,
        objects=(
            core_view(shield=4),
            *workers,
            *vanguards,
            enemy_view((9, 0)),
        ),
    )
    decide(turn)
    assert core_action_type(turn) == "SpawnAction"
    assert turn.plan.core_action.unit_type is UnitType.RANGER


def test_unit_heals_preserve_the_only_policy_approved_defender_price():
    from tactic import unit_heal_reserve

    workers = tuple(worker_view((20 + uid, 0), uid=uid) for uid in range(2, 16))
    vanguards = tuple(vanguard_view((uid, 2), uid=uid) for uid in range(16, 20))
    turn = make_turn(
        resources=12,
        objects=(
            core_view(),
            *workers,
            *vanguards,
            enemy_view((9, 0)),
        ),
    )
    assert unit_heal_reserve(turn, danger=False) == 12


def test_defense_telemetry_reports_a_blocked_spawn_cell():
    memory = ScoutMemory()
    turn = make_turn(
        resources=10,
        objects=(
            core_view(shield=2),
            worker_view((0, 0), cargo=1, uid=2),
            enemy_view((2, 0)),
            terrain("OBSTACLE", [(-1, 0), (0, -1), (0, 1)]),
        ),
    )
    decide(turn, memory)
    assert memory.last_defense_status is not None
    assert "cell:blocked" in memory.last_defense_status
    assert "spawn_blockers:spawn_cell" in memory.last_defense_status
    assert "decision:wait" in memory.last_defense_status


def test_defense_telemetry_separates_spawn_blocker_from_surplus_repair():
    memory = ScoutMemory()
    turn = make_turn(
        resources=14,
        objects=(
            core_view(shield=2),
            vanguard_view((0, 0), hp=1, uid=4),
            worker_view((5, 0), uid=2),
            worker_view((5, 1), uid=3),
            enemy_view((2, 0)),
        ),
    )
    decide(turn, memory)
    assert action_type(turn.plan, 4) == "HealAction"
    assert core_action_type(turn) == "RepairShieldAction"
    assert memory.last_defense_status is not None
    assert "spawn_blockers:spawn_cell" in memory.last_defense_status
    assert "decision:repair_shield" in memory.last_defense_status


def test_visible_distant_hostile_starts_defender_reserve_before_danger():
    turn = make_turn(
        resources=8,
        objects=(
            core_view(shield=4),
            worker_view((3, 0), uid=2),
            enemy_view((5, 0)),
        ),
    )
    decide(turn)
    assert core_action_type(turn) == "WaitAction"


def test_recent_damage_keeps_defender_reserve_after_enemy_leaves_vision():
    memory = ScoutMemory(core_threat_until_tick=105)
    turn = make_turn(
        tick=100,
        resources=8,
        objects=(
            core_view(shield=4),
            worker_view((5, 0), uid=2),
        ),
    )
    decide(turn, memory)
    assert core_action_type(turn) == "WaitAction"
    assert memory.last_defense_status is not None
    assert "reserve:10" in memory.last_defense_status
    assert "spawn_blockers:funds=2" in memory.last_defense_status


def test_recent_damage_spawns_defender_after_enemy_leaves_vision_when_funded():
    memory = ScoutMemory(core_threat_until_tick=105)
    turn = make_turn(
        tick=100,
        resources=10,
        objects=(
            core_view(shield=4),
            worker_view((5, 0), uid=2),
        ),
    )
    decide(turn, memory)
    assert core_action_type(turn) == "SpawnAction"
    assert turn.plan.core_action.unit_type is UnitType.VANGUARD


def test_visible_distant_hostile_prioritizes_first_defender_over_worker():
    workers = tuple(worker_view((10 + uid, 0), uid=uid) for uid in range(2, 8))
    turn = make_turn(
        resources=10,
        objects=(
            core_view(shield=4),
            *workers,
            enemy_view((5, 0)),
        ),
    )
    decide(turn)
    assert core_action_type(turn) == "SpawnAction"
    assert turn.plan.core_action.unit_type is UnitType.VANGUARD


def test_visible_distant_hostile_preserves_defender_budget_from_unit_healing():
    turn = make_turn(
        resources=10,
        objects=(
            core_view(),
            worker_view((0, 0), hp=1, uid=2),
            enemy_view((5, 0)),
        ),
    )
    decide(turn)
    assert action_type(turn.plan, 2) != "HealAction"
    assert core_action_type(turn) == "SpawnAction"
    assert turn.plan.core_action.unit_type is UnitType.VANGUARD


def test_visible_distant_hostile_preserves_defender_budget_from_core_healing():
    turn = make_turn(
        resources=10,
        objects=(
            core_view(hp=4),
            worker_view((3, 0), uid=2),
            enemy_view((5, 0)),
        ),
    )
    decide(turn)
    assert core_action_type(turn) == "SpawnAction"
    assert turn.plan.core_action.unit_type is UnitType.VANGUARD


def test_critical_core_healing_can_spend_below_defender_reserve():
    memory = ScoutMemory()
    turn = make_turn(
        resources=8,
        objects=(
            core_view(hp=2),
            worker_view((4, 0), uid=2),
            enemy_view((2, 0)),
        ),
    )
    decide(turn, memory)
    assert core_action_type(turn) == "HealAction"
    assert memory.last_defense_status is not None
    assert "decision:heal:critical" in memory.last_defense_status


def test_defender_reserve_uses_population_adjusted_price():
    workers = tuple(
        worker_view((10 + uid, 10), uid=uid) for uid in range(2, 22)
    )
    turn = make_turn(
        resources=12,
        objects=(core_view(shield=4), *workers, enemy_view((5, 0))),
    )
    decide(turn)
    assert core_action_type(turn) == "WaitAction"


def test_distant_hostile_keeps_reserve_when_a_guard_already_exists():
    turn = make_turn(
        resources=8,
        objects=(
            core_view(shield=4),
            worker_view((3, 0), uid=2),
            vanguard_view((2, 2), uid=4),
            enemy_view((5, 0)),
        ),
    )
    decide(turn)
    assert core_action_type(turn) == "WaitAction"


def test_worker_production_only_spends_above_visible_hostile_reserve():
    objects = (
        core_view(),
        worker_view((3, 0), uid=2),
        vanguard_view((2, 2), uid=4),
        enemy_view((5, 0)),
    )
    saving_turn = make_turn(resources=14, objects=objects)
    decide(saving_turn)
    assert core_action_type(saving_turn) == "WaitAction"

    surplus_turn = make_turn(resources=15, objects=objects)
    decide(surplus_turn)
    assert core_action_type(surplus_turn) == "SpawnAction"
    assert surplus_turn.plan.core_action.unit_type is UnitType.WORKER


def test_noncombat_enemies_do_not_trigger_defender_reserve():
    for enemy in (
        enemy_view((5, 0), unit_type=UnitType.WORKER),
        enemy_view((5, 0), kind="CORE"),
    ):
        turn = make_turn(
            resources=8,
            objects=(core_view(shield=4), worker_view((3, 0), uid=2), enemy),
        )
        decide(turn)
        assert core_action_type(turn) == "RepairShieldAction"


def test_diagonal_ranger_fire_triggers_defender_production():
    turn = make_turn(
        resources=20,
        objects=(
            core_view(shield=4),
            worker_view((5, 0), uid=2),
            worker_view((5, 1), uid=3),
            worker_view((5, 2), uid=4),
            worker_view((5, 3), uid=5),
            enemy_view((3, 3), unit_type=UnitType.RANGER),
        ),
    )
    decide(turn)
    assert core_action_type(turn) == "SpawnAction"
    assert turn.plan.core_action.unit_type is UnitType.RANGER


def test_danger_spawns_defender():
    turn = make_turn(
        resources=20,
        objects=(
            core_view(),
            worker_view((3, 0)),
            worker_view((3, 1), uid=3),
            enemy_view((2, 0)),
        ),
    )
    decide(turn)
    assert core_action_type(turn) == "SpawnAction"
    assert turn.plan.core_action.unit_type is UnitType.RANGER


def test_projected_population_loss_uses_post_combat_capacity_for_defender():
    turn = make_turn(
        resources=12,
        objects=(
            core_view(),
            vanguard_view((0, 1), hp=1, uid=4),
            worker_view((5, 0), uid=2),
            worker_view((5, 1), uid=3),
            enemy_view((3, 0), uid=90, unit_type=UnitType.RANGER),
            enemy_view((0, 2), uid=91, unit_type=UnitType.VANGUARD),
        ),
    )
    decide(turn)
    assert core_action_type(turn) == "SpawnAction"
    assert turn.plan.core_action.unit_type is UnitType.VANGUARD


def test_projected_capacity_includes_friendly_one_step_combat_exposure():
    turn = make_turn(
        objects=(
            core_view(position=(10, 10)),
            worker_view((2, 0), hp=1, uid=2),
            worker_view((10, 11), uid=3),
            worker_view((10, 12), uid=4),
            enemy_view((0, 0), uid=90, unit_type=UnitType.VANGUARD),
        ),
    )
    turn.unit(U(2)).move(Direction.LEFT)
    assert projected_post_combat_capacity(
        turn,
        turn.visible_enemies,
        turn.obstacle_cells,
    ) == 10


def test_planned_move_into_ranger_fire_reduces_core_spawn_budget():
    turn = make_turn(
        resources=12,
        objects=(
            core_view(),
            vanguard_view((0, 2), hp=1, uid=4),
            worker_view((10, 0), uid=2),
            worker_view((10, 1), uid=3),
            enemy_view((3, 0), uid=90, unit_type=UnitType.RANGER),
        ),
    )
    decide(turn)
    assert action_type(turn.plan, 4) == "MoveAction"
    assert direction_of(turn.plan, 4) is Direction.RIGHT
    assert core_action_type(turn) == "SpawnAction"
    assert turn.plan.core_action.unit_type is UnitType.VANGUARD


def test_respawning_waits():
    turn = make_turn(
        status=PlayerStatus.RESPAWNING,
        respawn_at_tick=101,
        resources=0,
        objects=(worker_view((0, 0), uid=2),),
    )
    decide(turn)
    assert action_type(turn.plan, 2) == "WaitAction"
    assert turn.plan.core_action is None


def test_ranger_shoots_legal_target():
    turn = make_turn(
        resources=10,
        objects=(core_view(), ranger_view((0, 0)), enemy_view((2, 0))),
    )
    decide(turn)
    assert action_type(turn.plan, 3) == "ShootAction"


def test_ranger_does_not_shoot_through_obstacle():
    turn = make_turn(
        resources=10,
        objects=(
            core_view(),
            ranger_view((0, 0)),
            enemy_view((2, 0)),
            terrain("OBSTACLE", [(1, 0)]),
        ),
    )
    decide(turn)
    assert action_type(turn.plan, 3) != "ShootAction"
    assert action_type(turn.plan, 3) == "MoveAction"


def test_rangers_distribute_lethal_damage_between_targets():
    turn = make_turn(
        resources=5,
        objects=(
            core_view(position=(-5, 0)),
            ranger_view((0, 0), uid=3),
            ranger_view((0, 0), uid=4),
            enemy_view((2, 0), hp=1, uid=90),
            enemy_view((3, 0), hp=1, uid=91),
        ),
    )
    decide(turn)
    assert {
        turn.plan.unit_actions[U(3)].target_id,
        turn.plan.unit_actions[U(4)].target_id,
    } == {U(90), U(91)}


def test_vanguard_sweeps_adjacent_enemy():
    turn = make_turn(
        resources=10,
        objects=(core_view(), vanguard_view((1, 0)), enemy_view((2, 0))),
    )
    decide(turn)
    assert action_type(turn.plan, 4) == "SweepAction"
    assert direction_of(turn.plan, 4) == Direction.RIGHT


def test_worker_heals_at_core():
    turn = make_turn(
        resources=10,
        objects=(core_view(), worker_view((0, 0), hp=1)),
    )
    decide(turn)
    assert action_type(turn.plan, 2) == "HealAction"


def test_obstacle_aware_pathing():
    turn = make_turn(
        resources=10,
        objects=(
            core_view(),
            worker_view((2, 0)),
            terrain("RESOURCE", [(4, 0)]),
            terrain("OBSTACLE", [(3, 0)]),
        ),
    )
    decide(turn)
    action = turn.plan.unit_actions.get(U(2))
    assert action is not None
    assert action_type(turn.plan, 2) == "MoveAction"
    dx, dy = action.direction.delta
    assert (2 + dx, 0 + dy) != (3, 0)


def test_pathing_reaches_a_distant_scout_waypoint():
    from tactic import step_toward

    assert step_toward((0, 0), (200, 0), set()) == (1, 0)


def test_pathing_uses_partial_route_when_distant_astar_exceeds_budget():
    from tactic import step_toward

    step = step_toward((0, 0), (100_000, 0), {(1, 0)})
    assert step in {(0, -1), (0, 1)}


def test_deterministic_and_fresh_plans():
    first = make_turn(resources=10, objects=(core_view(), worker_view((3, 0))))
    second = make_turn(
        resources=10,
        objects=(core_view(uid=11), worker_view((3, 0), uid=12)),
        tick=101,
    )
    decide(first)
    decide(second)
    assert first.plan.tick == 100
    assert second.plan.tick == 101
    assert U(2) in first.plan.unit_actions
    assert U(2) not in second.plan.unit_actions


def test_shot_geometry():
    obstacles = frozenset()
    assert is_legal_shot((0, 0), (3, 3), obstacles)
    assert is_legal_shot((0, 0), (0, 3), obstacles)
    assert is_legal_shot((0, 0), (2, 1), obstacles) is False
    assert is_legal_shot((0, 0), (4, 0), obstacles) is False
    blocked = frozenset({(1, 1)})
    assert is_legal_shot((0, 0), (2, 2), blocked) is False
    assert is_legal_shot((0, 0), (3, 3), blocked) is False
    assert is_legal_shot((0, 0), (3, 3), frozenset({(1, 0)})) is True


def test_cargo_worker_moves_toward_full_core():
    turn = make_turn(
        resources=5,
        objects=(
            core_view(),
            worker_view((0, 0), uid=2),
            worker_view((5, 0), cargo=1, uid=3),
        ),
    )
    decide(turn)
    assert action_type(turn.plan, 3) == "MoveAction"
    assert direction_of(turn.plan, 3) == Direction.LEFT


def test_idle_worker_steps_off_crowded_core():
    turn = make_turn(
        resources=4,
        objects=(
            core_view(),
            worker_view((0, 0), uid=2),
            worker_view((5, 0), cargo=1, uid=3),
        ),
    )
    decide(turn)
    assert action_type(turn.plan, 2) == "MoveAction"
    dx, dy = turn.plan.unit_actions.get(U(2)).direction.delta
    assert (0 + dx, 0 + dy) != (0, 0)


def test_worker_scouts_when_no_resources_visible():
    turn = make_turn(
        resources=4,
        objects=(core_view(), worker_view((0, 5), uid=2)),
    )
    decide(turn)
    assert action_type(turn.plan, 2) == "MoveAction"


def test_scout_rotates_after_reaching_target():
    memory = ScoutMemory(offsets={str(U(2)): 0})
    target = scout_target(str(U(2)), (0, 0), memory)
    turn = make_turn(
        resources=4,
        objects=(core_view(), worker_view(target, uid=2)),
    )
    decide(turn, memory)
    assert memory.offsets[str(U(2))] == 1


# --- scouting: was confined to four fixed cells at radius six ---------------


def test_scout_spreads_workers_onto_different_rays():
    memory = ScoutMemory()
    first = scout_target(str(U(2)), (0, 0), memory)
    second = scout_target(str(U(3)), (0, 0), memory)
    assert first != second


def test_scout_widens_the_ring_after_a_full_lap():
    inner = scout_target(str(U(2)), (0, 0), ScoutMemory(offsets={str(U(2)): 0}))
    outer = scout_target(str(U(2)), (0, 0), ScoutMemory(offsets={str(U(2)): 8}))
    assert max(abs(outer[0]), abs(outer[1])) > max(abs(inner[0]), abs(inner[1]))


def test_scout_covers_many_distinct_waypoints():
    from tactic import SCOUT_OFFSETS, SCOUT_RING_COUNT

    total = len(SCOUT_OFFSETS) * SCOUT_RING_COUNT
    seen = {
        scout_target(str(U(2)), (0, 0), ScoutMemory(offsets={str(U(2)): n}))
        for n in range(total)
    }
    assert len(seen) == total


def test_full_sweep_wraps_inside_the_bounded_ring():
    from tactic import SCOUT_OFFSETS, SCOUT_RING_COUNT, advance_scout

    total = len(SCOUT_OFFSETS) * SCOUT_RING_COUNT
    memory = ScoutMemory(offsets={str(U(2)): total - 1})
    advance_scout(str(U(2)), memory)
    assert memory.offsets[str(U(2))] == 0
    assert memory.sweeps[str(U(2))] == 1


def test_scout_sweeps_repeat_without_expanding_past_the_tether():
    from tactic import SCOUT_MAX_DISTANCE, SCOUT_OFFSETS, SCOUT_RING_COUNT

    total = len(SCOUT_OFFSETS) * SCOUT_RING_COUNT
    inner = scout_target(str(U(2)), (0, 0), ScoutMemory(offsets={str(U(2)): 0}))
    outer = scout_target(str(U(2)), (0, 0), ScoutMemory(offsets={str(U(2)): total}))
    assert outer == inner
    assert abs(outer[0]) + abs(outer[1]) <= SCOUT_MAX_DISTANCE


def test_walled_in_worker_advances_instead_of_stalling():
    memory = ScoutMemory(offsets={str(U(2)): 0})
    turn = make_turn(
        resources=4,
        objects=(
            core_view(),
            worker_view((10, 10), uid=2),
            terrain("OBSTACLE", [(9, 10), (11, 10), (10, 9), (10, 11)]),
        ),
    )
    decide(turn, memory)
    assert action_type(turn.plan, 2) == "WaitAction"
    assert memory.offsets[str(U(2))] == 8


def test_scout_progress_survives_a_restart(tmp_path):
    path = tmp_path / "scout_state.json"
    saved = ScoutMemory(
        offsets={str(U(2)): 7},
        sweeps={str(U(2)): 2},
        known_obstacles={(4, 5), (6, 7)},
        known_resources={(8, 9), (10, 11)},
        resource_assignments={str(U(2)): (8, 9)},
        core_threat_until_tick=123,
        path=path,
        dirty=True,
    )
    saved.save()
    restored = ScoutMemory(path=path)
    restored.load()
    assert restored.offsets == {str(U(2)): 7}
    assert restored.sweeps == {str(U(2)): 2}
    assert restored.known_obstacles == {(4, 5), (6, 7)}
    assert restored.known_resources == {(8, 9), (10, 11)}
    assert restored.resource_assignments == {str(U(2)): (8, 9)}
    assert restored.core_threat_until_tick == 123


def test_legacy_state_without_obstacles_still_loads(tmp_path):
    path = tmp_path / "scout_state.json"
    path.write_text('{"offsets": {}}')
    memory = ScoutMemory(path=path)
    memory.load()
    assert memory.known_obstacles == set()


def test_unusable_state_file_does_not_crash(tmp_path):
    path = tmp_path / "scout_state.json"
    path.write_text("not json")
    memory = ScoutMemory(path=path)
    memory.load()
    assert memory.offsets == {}


# --- resource scheduling and remembered terrain -----------------------------


def test_nearest_worker_claims_resource_regardless_of_uuid_order():
    turn = make_turn(
        resources=5,
        objects=(
            core_view(),
            worker_view((0, 0), uid=2),
            worker_view((8, 0), uid=3),
            terrain("RESOURCE", [(10, 0)]),
        ),
    )
    decide(turn)
    assert action_type(turn.plan, 3) == "MoveAction"
    assert direction_of(turn.plan, 3) == Direction.RIGHT
    assert turn.plan.unit_actions[U(3)].direction == Direction.RIGHT


def test_worker_on_assigned_resource_harvests():
    turn = make_turn(
        resources=5,
        objects=(
            core_view(),
            worker_view((10, 0), uid=3),
            worker_view((0, 0), uid=2),
            terrain("RESOURCE", [(10, 0)]),
        ),
    )
    decide(turn)
    assert action_type(turn.plan, 3) == "HarvestAction"


def test_resource_assignment_tie_breaks_by_worker_id():
    from tactic import assign_resource_targets

    turn = make_turn(
        objects=(core_view(), worker_view((4, 0), uid=2), worker_view((6, 0), uid=3)),
    )
    targets = assign_resource_targets(turn.workers, {(5, 0)})
    assert targets == {U(2): (5, 0)}


def test_resource_map_remembers_a_node_after_it_leaves_view():
    memory = ScoutMemory()
    discovery = make_turn(
        resources=5,
        objects=(core_view(), worker_view((0, 0), uid=2), terrain("RESOURCE", [(8, 0)])),
    )
    decide(discovery, memory)
    assert memory.known_resources == {(8, 0)}

    later = make_turn(
        resources=5,
        objects=(core_view(), worker_view((0, 0), uid=2)),
        tick=101,
    )
    decide(later, memory)
    assert action_type(later.plan, 2) == "MoveAction"
    assert direction_of(later.plan, 2) == Direction.RIGHT


def test_worker_on_nonvisible_remembered_node_forgets_it_and_scouts():
    memory = ScoutMemory(known_resources={(8, 0)})
    turn = make_turn(
        resources=5,
        objects=(core_view(), worker_view((8, 0), uid=2)),
    )
    decide(turn, memory)
    assert (8, 0) not in memory.known_resources
    assert action_type(turn.plan, 2) == "MoveAction"


def test_resource_memory_survives_when_outside_friendly_vision():
    memory = ScoutMemory(known_resources={(8, 0)})
    turn = make_turn(
        resources=5,
        objects=(core_view(), worker_view((0, 0), uid=2)),
    )
    decide(turn, memory)
    assert (8, 0) in memory.known_resources
    assert action_type(turn.plan, 2) == "MoveAction"


def test_resource_memory_survives_when_obstacle_hides_visible_range():
    memory = ScoutMemory(known_resources={(3, 0)})
    turn = make_turn(
        resources=5,
        objects=(
            core_view(),
            worker_view((0, 0), uid=2),
            terrain("OBSTACLE", [(1, 0)]),
        ),
    )
    decide(turn, memory)
    assert (3, 0) in memory.known_resources


def test_core_vision_clears_missing_visible_resource():
    memory = ScoutMemory(known_resources={(3, 0)})
    turn = make_turn(
        resources=5,
        objects=(core_view(), worker_view((0, 5), uid=2)),
    )
    decide(turn, memory)
    assert (3, 0) not in memory.known_resources





def test_harvest_failure_removes_remembered_node():
    memory = ScoutMemory(known_resources={(2, 0)})
    turn = make_turn(
        resources=5,
        objects=(core_view(), worker_view((2, 0), uid=2)),
        events=(
            ResolutionEvent(
                event_id=U(63),
                tick=99,
                event_type="HARVEST_FAILED",
                reason_code="NODE_EXHAUSTED",
                actor_id=U(2),
                position=(2, 0),
            ),
        ),
    )
    decide(turn, memory)
    assert (2, 0) not in memory.known_resources
    assert (2, 0) in memory.depleted
    assert action_type(turn.plan, 2) != "HarvestAction"


def test_cargo_full_harvest_failure_does_not_erase_the_resource():
    memory = ScoutMemory(
        known_resources={(2, 0)},
        resource_last_seen={(2, 0): 99},
    )
    turn = make_turn(
        resources=5,
        objects=(
            core_view(),
            worker_view((2, 0), cargo=1, uid=2),
            terrain("RESOURCE", [(2, 0)]),
        ),
        events=(
            ResolutionEvent(
                event_id=U(65),
                tick=99,
                event_type="HARVEST_FAILED",
                reason_code="CARGO_FULL",
                actor_id=U(2),
                position=(2, 0),
            ),
        ),
    )
    decide(turn, memory)
    assert (2, 0) in memory.known_resources
    assert (2, 0) not in memory.depleted


def test_scout_grid_targets_are_distinct_for_idle_workers():
    from tactic import scout_coverage_target

    memory = ScoutMemory(scout_seen={(0, 0): 100})
    first = worker_view((0, 0), uid=2)
    second = worker_view((0, 1), uid=3)
    claimed = set()
    first_target = scout_coverage_target(first, (0, 0), memory, claimed, 101)
    second_target = scout_coverage_target(second, (0, 0), memory, claimed, 101)
    assert first_target is not None
    assert second_target is not None
    assert memory.scout_targets[str(first.id)] != memory.scout_targets[str(second.id)]


def test_scout_grid_targets_prefer_stable_worker_sectors():
    from tactic import (
        ensure_scout_sector_slots,
        scout_coverage_target,
        scout_sector,
    )

    memory = ScoutMemory()
    claimed: set[tuple[int, int]] = set()
    workers = tuple(worker_view((0, 0), uid=uid) for uid in range(2, 6))
    ensure_scout_sector_slots({str(worker.id) for worker in workers}, memory)
    targets = [
        scout_coverage_target(worker, (0, 0), memory, claimed, 101)
        for worker in workers
    ]

    assert all(target is not None for target in targets)
    assert len(set(targets)) == len(targets)
    assert all(
        scout_sector(target, (0, 0)) == memory.scout_sector_slots[str(worker.id)]
        for worker, target in zip(workers, targets)
    )
    assert set(memory.scout_sector_slots.values()) == {0, 1, 2, 3}


def test_scout_sector_outranks_unseen_cells_in_other_sectors():
    from tactic import (
        SCOUT_MAX_DISTANCE,
        scout_coverage_target,
        scout_grid_disc,
        scout_sector,
        scout_sector_for_worker,
    )

    worker = worker_view((0, 0), uid=2)
    preferred = scout_sector_for_worker(str(worker.id))
    memory = ScoutMemory(
        scout_sector_slots={str(worker.id): preferred},
        scout_seen={
            cell: 100
            for cell in scout_grid_disc((0, 0), SCOUT_MAX_DISTANCE)
            if scout_sector(
                (cell[0] * 3 + 1, cell[1] * 3 + 1),
                (0, 0),
            )
            == preferred
        }
    )

    target = scout_coverage_target(worker, (0, 0), memory, set(), 101)
    assert target is not None
    assert scout_sector(target, (0, 0)) == preferred


def test_recently_seen_scout_cell_is_not_preferred():
    from tactic import scout_coverage_target

    memory = ScoutMemory(scout_seen={(1, 0): 100})
    worker = worker_view((0, 0), uid=2)
    target = scout_coverage_target(worker, (0, 0), memory, set(), 101)
    assert target is not None
    assert memory.scout_targets[str(worker.id)] != (1, 0)


def test_scout_loop_releases_previous_coverage_target():
    from tactic import scout_coverage_target

    worker = worker_view((0, 0), uid=2)
    memory = ScoutMemory(scout_targets={str(worker.id): (1, 0)})
    for position in ((0, 0), (1, 0), (0, 0), (1, 0), (0, 0), (1, 0)):
        memory.record_position(str(worker.id), position)
    claimed = set()
    target = scout_coverage_target(worker, (0, 0), memory, claimed, 100)
    assert target is not None
    assert memory.scout_targets[str(worker.id)] != (1, 0)
    assert memory.scout_seen[(1, 0)] == 100


def test_scout_coverage_state_survives_restart(tmp_path):
    path = tmp_path / "scout_state.json"
    saved = ScoutMemory(
        scout_seen={(1, -1): 42},
        scout_targets={str(U(2)): (2, 0)},
        scout_sector_slots={str(U(2)): 3},
        scout_roles={str(U(2)): "remote"},
        scout_positions={str(U(2)): [(0, 0), (1, 0)]},
        last_move_destinations={str(U(2)): (1, 0)},
        path=path,
        dirty=True,
    )
    saved.save()
    restored = ScoutMemory(path=path)
    restored.load()
    assert restored.scout_seen == {(1, -1): 42}
    assert restored.scout_targets == {str(U(2)): (2, 0)}
    assert restored.scout_sector_slots == {str(U(2)): 3}
    assert restored.scout_roles == {str(U(2)): "remote"}
    assert restored.scout_positions == {str(U(2)): [(0, 0), (1, 0)]}
    assert restored.last_move_destinations == {str(U(2)): (1, 0)}


def test_scout_sector_slots_are_balanced_and_released_after_worker_death():
    from tactic import ensure_scout_sector_slots

    workers = {str(U(uid)) for uid in range(2, 6)}
    memory = ScoutMemory()
    ensure_scout_sector_slots(workers, memory)
    assert set(memory.scout_sector_slots.values()) == {0, 1, 2, 3}

    survivor = str(U(2))
    survivor_sector = memory.scout_sector_slots[survivor]
    memory.prune_workers({survivor})
    assert memory.scout_sector_slots == {survivor: survivor_sector}

    replacement = str(U(6))
    ensure_scout_sector_slots({survivor, replacement}, memory)
    assert memory.scout_sector_slots[survivor] == survivor_sector
    assert memory.scout_sector_slots[replacement] != survivor_sector


def test_scout_roles_split_workers_deterministically_between_local_and_remote():
    from tactic import ensure_scout_roles

    worker_ids = {str(U(uid)) for uid in range(2, 16)}
    first = ScoutMemory()
    second = ScoutMemory()
    ensure_scout_roles(worker_ids, first)
    ensure_scout_roles(reversed(sorted(worker_ids)), second)

    assert first.scout_roles == second.scout_roles
    assert Counter(first.scout_roles.values()) == Counter(local=8, remote=6)


def test_scout_roles_have_separate_local_and_remote_target_discs():
    from tactic import ensure_scout_roles, scout_coverage_target, scout_radius_for_worker

    turn = make_turn(
        objects=(
            core_view(),
            *(worker_view((0, uid), uid=uid) for uid in range(2, 16)),
        ),
        tick=100,
    )
    memory = ScoutMemory()
    ensure_scout_roles({str(worker.id) for worker in turn.workers}, memory)
    claimed = set()
    distances = {}
    for worker in turn.workers:
        worker_id = str(worker.id)
        radius = scout_radius_for_worker(worker_id, memory)
        target = scout_coverage_target(
            worker,
            (0, 0),
            memory,
            claimed,
            turn.tick,
            max_distance=radius,
            min_distance=41 if memory.scout_roles[worker_id] == "remote" else 0,
        )
        assert target is not None
        distances[memory.scout_roles[worker_id], worker_id] = sum(map(abs, target))

    assert max(
        distance for (role, _), distance in distances.items() if role == "local"
    ) <= 40
    assert min(
        distance for (role, _), distance in distances.items() if role == "remote"
    ) > 40


def move_destinations(turn):
    destinations = []
    for unit in turn.units:
        action = turn.plan.unit_actions.get(unit.id)
        if type(action).__name__ != "MoveAction":
            continue
        dx, dy = action.direction.delta
        destinations.append((unit.position[0] + dx, unit.position[1] + dy))
    return destinations


def test_movement_reservations_prevent_duplicate_destinations():
    from tactic import MovementReservations, move_or_wait

    turn = make_turn(
        objects=(core_view(), worker_view((0, 0), uid=2), worker_view((0, 2), uid=3)),
    )
    reservations = MovementReservations()
    assert move_or_wait(turn.unit(U(2)), (2, 0), set(), reservations)
    assert move_or_wait(turn.unit(U(3)), (1, 0), set(), reservations) is False
    assert len(reservations.destinations) == 1
    assert len(move_destinations(turn)) == len(set(move_destinations(turn)))
    assert (1, 0) in reservations.destinations


def test_cargo_worker_reserves_shared_step_before_scout():
    turn = make_turn(
        resources=5,
        objects=(
            core_view(),
            worker_view((2, 0), cargo=1, uid=2),
            worker_view((1, 1), uid=3),
        ),
    )
    decide(turn)
    assert action_type(turn.plan, 2) == "MoveAction"
    assert direction_of(turn.plan, 2) == Direction.LEFT
    assert len(move_destinations(turn)) == len(set(move_destinations(turn)))


def test_resource_miner_reserves_shared_step_before_scout():
    turn = make_turn(
        resources=5,
        objects=(
            core_view(),
            worker_view((0, 0), uid=2),
            worker_view((1, 1), uid=3),
            terrain("RESOURCE", [(2, 0)]),
        ),
    )
    decide(turn)
    assert action_type(turn.plan, 2) == "MoveAction"
    assert direction_of(turn.plan, 2) == Direction.RIGHT
    assert len(move_destinations(turn)) == len(set(move_destinations(turn)))


def test_contested_cell_is_avoided_on_the_following_tick():
    memory = ScoutMemory(last_move_destinations={str(U(2)): (1, 0)})
    event = ResolutionEvent(
        event_id=U(64),
        tick=99,
        event_type="UNIT_MOVE_FAILED",
        reason_code="MOVE_CONTESTED",
        actor_id=U(2),
        # Failed movement events expose the unchanged origin, not destination.
        position=(0, 0),
    )
    turn = make_turn(
        objects=(core_view(), worker_view((0, 0), uid=2)),
        events=(event,),
        tick=100,
    )
    decide(turn, memory)
    assert (1, 0) in memory.contested
    assert (0, 0) not in memory.contested
    assert (1, 0) not in move_destinations(turn)


def test_contested_cell_backoff_expires():
    memory = ScoutMemory(contested={(1, 0): 100})
    turn = make_turn(
        objects=(core_view(), worker_view((0, 0), uid=2)),
        tick=100,
    )
    decide(turn, memory)
    assert memory.contested == {}


# --- pathing: enemy cells were never actually blocked ----------------------


def test_pathing_routes_around_enemy_cells():
    turn = make_turn(
        resources=5,
        objects=(core_view(), worker_view((2, 0), cargo=1, uid=2), enemy_view((1, 0))),
    )
    decide(turn)
    action = turn.plan.unit_actions.get(U(2))
    assert action_type(turn.plan, 2) == "MoveAction"
    dx, dy = action.direction.delta
    assert (2 + dx, 0 + dy) != (1, 0)


def test_combat_unit_paths_through_a_single_friendly_cell():
    turn = make_turn(
        resources=5,
        objects=(
            core_view(),
            ranger_view((3, 0), uid=3),
            worker_view((4, 0), uid=2),
            enemy_view((7, 0)),
        ),
    )
    decide(turn)
    assert action_type(turn.plan, 3) == "MoveAction"
    assert direction_of(turn.plan, 3) == Direction.RIGHT


# --- budgets: deposits overflowed and heals were unbounded ------------------


def test_second_depositor_waits_when_storage_is_full():
    turn = make_turn(
        resources=9,
        objects=(
            core_view(),
            worker_view((0, 0), cargo=3, uid=2),
            worker_view((0, 0), cargo=3, uid=3),
        ),
    )
    decide(turn)
    assert action_type(turn.plan, 2) == "DepositAction"
    assert action_type(turn.plan, 3) == "WaitAction"


def test_healing_stops_at_the_reserve():
    turn = make_turn(
        resources=3,
        objects=(
            core_view(),
            worker_view((0, 0), hp=1, uid=2),
            worker_view((0, 0), hp=1, uid=3),
        ),
    )
    decide(turn)
    assert action_type(turn.plan, 2) == "HealAction"
    assert action_type(turn.plan, 3) != "HealAction"


def test_affordable_drops_the_reserve_when_capacity_forbids_it():
    assert affordable(12, TickBudget(resources=20, space=0), capacity=30) is True
    assert affordable(12, TickBudget(resources=12, space=0), capacity=10) is True
    assert affordable(12, TickBudget(resources=11, space=0), capacity=10) is False


def test_low_population_core_can_still_field_a_defender():
    turn = make_turn(
        resources=10,
        objects=(core_view(), worker_view((3, 0), uid=2), enemy_view((2, 0))),
    )
    decide(turn)
    assert core_action_type(turn) == "SpawnAction"
    assert turn.plan.core_action.unit_type is UnitType.VANGUARD


# --- Champion Beacon -------------------------------------------------------


def test_unit_picks_up_the_beacon_it_stands_on():
    turn = make_turn(
        resources=5,
        objects=(core_view(), worker_view((3, 0), uid=2), worker_view((0, 1), uid=3)),
        beacon=ChampionBeacon(position=(3, 0), status=BeaconStatus.GROUND),
    )
    decide(turn)
    assert action_type(turn.plan, 2) == "PickupBeaconAction"


def test_our_carrier_brings_the_beacon_home():
    turn = make_turn(
        resources=5,
        objects=(core_view(), vanguard_view((5, 0), uid=4), worker_view((0, 1), uid=2)),
        beacon=ChampionBeacon(
            position=(5, 0), status=BeaconStatus.CARRIED, carrier_id=U(4)
        ),
    )
    decide(turn)
    assert action_type(turn.plan, 4) == "MoveAction"
    assert direction_of(turn.plan, 4) == Direction.LEFT


def test_lone_unit_keeps_working_instead_of_chasing_the_beacon():
    turn = make_turn(
        resources=5,
        objects=(core_view(), worker_view((1, 0), uid=2)),
        beacon=ChampionBeacon(position=(3, 0), status=BeaconStatus.GROUND),
    )
    decide(turn)
    assert action_type(turn.plan, 2) != "PickupBeaconAction"


def test_only_guard_is_not_diverted_to_pick_up_the_beacon():
    turn = make_turn(
        resources=5,
        objects=(
            core_view(),
            vanguard_view((3, 0), uid=4),
            worker_view((20, 0), uid=2),
            worker_view((20, 1), uid=3),
            worker_view((20, 2), uid=5),
            worker_view((20, 3), uid=6),
        ),
        beacon=ChampionBeacon(position=(3, 0), status=BeaconStatus.GROUND),
    )
    decide(turn)
    assert action_type(turn.plan, 4) != "PickupBeaconAction"


def test_completed_roster_repairs_beacon_shield_above_five():
    workers = tuple(worker_view((20 + uid, 0), uid=uid) for uid in range(2, 16))
    defenders = (
        *(vanguard_view((uid, 2), uid=uid) for uid in range(16, 20)),
        *(ranger_view((uid, 3), uid=uid) for uid in range(20, 22)),
    )
    turn = make_turn(
        resources=10,
        objects=(core_view(shield=5), *workers, *defenders),
        beacon=ChampionBeacon(
            position=defenders[0].position,
            status=BeaconStatus.CARRIED,
            carrier_id=defenders[0].id,
        ),
    )
    decide(turn)
    assert core_action_type(turn) == "RepairShieldAction"


def test_ranger_prioritises_immediate_core_threat_over_beacon_carrier():
    turn = make_turn(
        resources=5,
        objects=(
            core_view(),
            ranger_view((0, 0), uid=3),
            worker_view((0, 5), uid=2),
            enemy_view((1, 0), uid=90),
            enemy_view((3, 0), uid=91),
        ),
        beacon=ChampionBeacon(
            position=(3, 0), status=BeaconStatus.CARRIED, carrier_id=U(91)
        ),
    )
    decide(turn)
    assert action_type(turn.plan, 3) == "ShootAction"
    assert turn.plan.unit_actions[U(3)].target_id == U(90)


def test_ranger_prioritises_beacon_carrier_without_a_core_threat():
    turn = make_turn(
        resources=5,
        objects=(
            core_view(position=(-5, 0)),
            ranger_view((0, 0), uid=3),
            worker_view((-5, 5), uid=2),
            enemy_view((1, 0), uid=90),
            enemy_view((3, 0), uid=91),
        ),
        beacon=ChampionBeacon(
            position=(3, 0), status=BeaconStatus.CARRIED, carrier_id=U(91)
        ),
    )
    decide(turn)
    assert action_type(turn.plan, 3) == "ShootAction"
    assert turn.plan.unit_actions[U(3)].target_id == U(91)


# --- resolution events -----------------------------------------------------


def test_failed_harvest_marks_the_cell_depleted():
    memory = ScoutMemory()
    turn = make_turn(
        resources=5,
        objects=(core_view(), worker_view((2, 0), uid=2)),
        events=(
            ResolutionEvent(
                event_id=U(60),
                tick=99,
                event_type="HARVEST_FAILED",
                reason_code="NODE_EXHAUSTED",
                actor_id=U(2),
                position=(2, 0),
            ),
        ),
    )
    decide(turn, memory)
    assert (2, 0) in memory.depleted
    assert action_type(turn.plan, 2) != "HarvestAction"


def test_visible_cargo_pile_survives_resource_depleted_event():
    memory = ScoutMemory(known_resources={(2, 0)})
    turn = make_turn(
        resources=5,
        objects=(
            core_view(),
            worker_view((2, 0), uid=2),
            terrain("RESOURCE", [(2, 0)]),
        ),
        events=(
            ResolutionEvent(
                event_id=U(61),
                tick=99,
                event_type="HARVEST_FAILED",
                reason_code="RESOURCE_DEPLETED",
                actor_id=U(2),
                position=(2, 0),
            ),
        ),
    )
    decide(turn, memory)
    assert (2, 0) not in memory.depleted
    assert (2, 0) in memory.known_resources
    assert action_type(turn.plan, 2) == "HarvestAction"


def test_depleted_cells_expire():
    memory = ScoutMemory(depleted={(2, 0): 50})
    turn = make_turn(
        resources=5,
        objects=(core_view(), worker_view((2, 0), uid=2), terrain("RESOURCE", [(2, 0)])),
        tick=100,
    )
    decide(turn, memory)
    assert (2, 0) not in memory.depleted
    assert action_type(turn.plan, 2) == "HarvestAction"


def test_events_are_counted_for_the_log():
    memory = ScoutMemory()
    turn = make_turn(
        resources=5,
        objects=(core_view(), worker_view((3, 0), uid=2)),
        events=(
            ResolutionEvent(event_id=U(61), tick=99, event_type="HARVEST_SUCCEEDED"),
            ResolutionEvent(event_id=U(62), tick=99, event_type="HARVEST_SUCCEEDED"),
        ),
    )
    decide(turn, memory)
    assert memory.last_events["HARVEST_SUCCEEDED"] == 2


# --- exact visibility -------------------------------------------------------


def test_visibility_uses_manhattan_radius():
    from tactic import cell_visible_to_friendly

    turn = make_turn(
        objects=(core_view(), worker_view((50, 50), uid=2)),
    )
    assert cell_visible_to_friendly(turn, (3, 2)) is True
    assert cell_visible_to_friendly(turn, (4, 4)) is False


def test_supercover_visibility_blocks_both_corner_cells():
    from tactic import _line_cells, cell_visible_to_friendly

    cells = _line_cells((0, 0), (2, 2))
    assert (1, 0) in cells
    assert (0, 1) in cells
    assert (2, 1) in cells
    assert (1, 2) in cells

    blocked_x = make_turn(
        objects=(
            core_view(),
            worker_view((50, 50), uid=2),
            terrain("OBSTACLE", [(1, 0)]),
        ),
    )
    blocked_y = make_turn(
        objects=(
            core_view(),
            worker_view((50, 50), uid=2),
            terrain("OBSTACLE", [(0, 1)]),
        ),
    )
    assert cell_visible_to_friendly(blocked_x, (2, 2)) is False
    assert cell_visible_to_friendly(blocked_y, (2, 2)) is False


# --- route-aware resource memory and assignment ----------------------------


def test_resource_assignment_prefers_the_shorter_route_around_walls():
    from tactic import assign_resource_targets

    turn = make_turn(
        objects=(
            core_view((20, 20)),
            worker_view((0, 0), uid=2),
            worker_view((4, 5), uid=3),
        ),
    )
    wall = {(1, y) for y in range(-3, 3)}
    targets = assign_resource_targets(turn.workers, {(4, 0)}, wall)
    assert targets == {U(3): (4, 0)}


def test_resource_assignment_skips_an_enemy_occupied_node():
    from tactic import assign_resource_targets

    turn = make_turn(objects=(core_view(), worker_view((0, 0), uid=2)))
    assert assign_resource_targets(turn.workers, {(2, 0)}, {(2, 0)}) == {}


def test_active_far_resource_is_remembered_but_uneconomic_assignment_is_dropped():
    target = (100, 0)
    memory = ScoutMemory(
        known_resources={target},
        resource_last_seen={target: 0},
        resource_assignments={str(U(2)): target},
    )
    turn = make_turn(
        tick=100,
        resources=0,
        objects=(core_view(), worker_view((50, 0), uid=2)),
    )
    decide(turn, memory)
    assert target in memory.known_resources
    assert str(U(2)) not in memory.resource_assignments
    assert memory.last_intents["recalling"] == 1


def test_unassigned_resource_still_expires_at_the_base_ttl():
    target = (100, 0)
    memory = ScoutMemory(
        known_resources={target},
        resource_last_seen={target: 0},
    )
    turn = make_turn(
        tick=100,
        resources=0,
        objects=(core_view(), worker_view((50, 0), uid=2)),
    )
    decide(turn, memory)
    assert target not in memory.known_resources


def test_dead_worker_state_is_pruned_from_every_worker_map():
    alive = str(U(2))
    dead = str(U(99))
    memory = ScoutMemory(
        offsets={alive: 1, dead: 2},
        sweeps={alive: 1, dead: 2},
        scout_targets={alive: (1, 0), dead: (2, 0)},
        scout_positions={alive: [(0, 0)], dead: [(1, 1)]},
        resource_assignments={dead: (30, 0)},
    )
    turn = make_turn(resources=0, objects=(core_view(), worker_view((3, 0), uid=2)))
    decide(turn, memory)
    for mapping in (
        memory.offsets,
        memory.sweeps,
        memory.scout_targets,
        memory.scout_positions,
        memory.resource_assignments,
    ):
        assert dead not in mapping


# --- two-phase Core planning and production --------------------------------


def test_same_tick_deposit_funds_core_healing():
    turn = make_turn(
        resources=0,
        objects=(core_view(hp=3), worker_view((0, 0), cargo=2, uid=2)),
    )
    decide(turn)
    assert action_type(turn.plan, 2) == "DepositAction"
    assert core_action_type(turn) == "HealAction"


def test_same_tick_deposit_funds_shield_repair():
    turn = make_turn(
        resources=0,
        objects=(core_view(shield=4), worker_view((0, 0), cargo=1, uid=2)),
    )
    decide(turn)
    assert action_type(turn.plan, 2) == "DepositAction"
    assert core_action_type(turn) == "RepairShieldAction"


def test_vanguard_heal_reserves_its_maximum_surviving_combat_cost():
    from tactic import plan_unit_heals

    turn = make_turn(
        resources=5,
        objects=(core_view(), vanguard_view((0, 0), hp=3, uid=4)),
    )
    budget = TickBudget(resources=5, space=0)
    healed = plan_unit_heals(turn, budget, danger=False, intents=Counter())
    assert healed == {U(4)}
    assert budget.resources == 2


def test_core_does_not_spawn_while_a_unit_will_remain_on_its_cell():
    turn = make_turn(
        resources=9,
        objects=(core_view(), worker_view((0, 0), cargo=1, uid=2)),
    )
    decide(turn)
    assert action_type(turn.plan, 2) == "DepositAction"
    assert core_action_type(turn) == "WaitAction"


def test_critical_core_heals_before_reactive_spawning():
    turn = make_turn(
        resources=20,
        objects=(
            core_view(hp=1),
            worker_view((3, 0), uid=2),
            enemy_view((2, 0)),
        ),
    )
    decide(turn)
    assert core_action_type(turn) == "HealAction"


def test_danger_preserves_defender_budget_instead_of_healing_a_worker():
    turn = make_turn(
        resources=10,
        objects=(
            core_view(),
            worker_view((0, 0), hp=1, uid=2),
            worker_view((3, 0), uid=3),
            enemy_view((2, 0)),
        ),
    )
    decide(turn)
    assert action_type(turn.plan, 2) != "HealAction"
    assert core_action_type(turn) == "SpawnAction"
    assert turn.plan.core_action.unit_type is UnitType.VANGUARD


def test_five_workers_trigger_a_proactive_guard():
    turn = make_turn(
        resources=10,
        objects=(
            core_view(),
            *(worker_view((5 + uid, 0), uid=uid) for uid in range(2, 7)),
        ),
    )
    decide(turn)
    assert core_action_type(turn) == "SpawnAction"
    assert turn.plan.core_action.unit_type is UnitType.VANGUARD


def test_economy_waits_for_repair_reserve_after_the_first_guard():
    turn = make_turn(
        resources=5,
        objects=(
            core_view(),
            *(worker_view((5 + uid, 0), uid=uid) for uid in range(2, 7)),
            vanguard_view((0, 3), uid=20),
        ),
    )
    decide(turn)
    assert core_action_type(turn) == "WaitAction"


def test_economy_expands_after_the_first_guard_with_reserve():
    turn = make_turn(
        resources=7,
        objects=(
            core_view(),
            *(worker_view((5 + uid, 0), uid=uid) for uid in range(2, 7)),
            vanguard_view((0, 3), uid=20),
        ),
    )
    decide(turn)
    assert core_action_type(turn) == "SpawnAction"
    assert turn.plan.core_action.unit_type is UnitType.WORKER


def test_four_workers_build_the_first_guard_before_expanding():
    turn = make_turn(
        resources=10,
        objects=(
            core_view(),
            *(worker_view((5 + uid, 0), uid=uid) for uid in range(2, 6)),
        ),
    )
    decide(turn)
    assert core_action_type(turn) == "SpawnAction"
    assert turn.plan.core_action.unit_type is UnitType.VANGUARD


def test_base_price_roster_keeps_adding_defenders_after_worker_cap():
    turn = make_turn(
        resources=12,
        objects=(
            core_view(),
            *(worker_view((20 + uid, 0), uid=uid) for uid in range(2, 16)),
            vanguard_view((0, 3), uid=30),
            ranger_view((0, 4), uid=31),
            vanguard_view((0, 5), uid=32),
            ranger_view((0, 6), uid=33),
        ),
    )
    decide(turn)
    assert core_action_type(turn) == "SpawnAction"
    assert turn.plan.core_action.unit_type is UnitType.VANGUARD


def test_idle_combat_unit_guards_instead_of_waiting_on_the_core():
    turn = make_turn(
        resources=0,
        objects=(core_view(), worker_view((4, 0), uid=2), vanguard_view((0, 0), uid=4)),
    )
    decide(turn)
    assert action_type(turn.plan, 4) == "MoveAction"


# --- conservative Core migration -------------------------------------------


def economic_migration_objects(*, moving=False, cargo_distance=None):
    core = (
        core_view(
            state=CoreState.MOVING,
            move_direction=Direction.RIGHT,
            move_progress=2,
            move_required_ticks=4,
            destination=(1, 0),
        )
        if moving
        else core_view()
    )
    workers = [
        worker_view((10 + index, 0), uid=2 + index)
        for index in range(5)
    ]
    if cargo_distance is not None:
        workers[0] = worker_view((cargo_distance, 0), cargo=1, uid=2)
    return (core, *workers)


def test_core_does_not_chase_one_stale_distant_resource():
    memory = ScoutMemory(
        known_resources={(20, 0)},
        resource_last_seen={(20, 0): 40},
    )
    turn = make_turn(
        resources=0,
        objects=economic_migration_objects(),
    )
    decide(turn, memory)
    assert core_action_type(turn) == "WaitAction"


def test_core_relocates_toward_multiple_active_resource_routes():
    memory = ScoutMemory(
        known_resources={(20, 0), (20, 2)},
        resource_last_seen={(20, 0): 100, (20, 2): 100},
    )
    turn = make_turn(
        resources=0,
        objects=economic_migration_objects(),
    )
    decide(turn, memory)
    assert core_action_type(turn) == "StartMoveAction"
    assert turn.plan.core_action.direction is Direction.RIGHT


def test_unit_healing_blocks_same_tick_core_migration():
    memory = ScoutMemory(
        known_resources={(20, 0), (20, 2)},
        resource_last_seen={(20, 0): 100, (20, 2): 100},
    )
    turn = make_turn(
        resources=5,
        objects=(
            core_view(),
            worker_view((0, 0), hp=1, uid=2),
            worker_view((10, 0), uid=3),
            worker_view((11, 0), uid=4),
            worker_view((12, 0), uid=5),
            worker_view((13, 0), uid=6),
        ),
    )
    decide(turn, memory)
    assert action_type(turn.plan, 2) == "HealAction"
    assert core_action_type(turn) == "WaitAction"
    assert memory.last_migration_hold == "unit_healing"


def test_empty_core_intercepts_nearest_cargo_instead_of_route_centroid():
    turn = make_turn(
        resources=0,
        objects=(
            core_view(),
            worker_view((-100, 0), cargo=1, uid=2),
            worker_view((20, 0), cargo=1, uid=3),
            worker_view((30, 0), uid=4),
            worker_view((31, 0), uid=5),
        ),
    )
    memory = ScoutMemory()
    decide(turn, memory)
    assert core_action_type(turn) == "StartMoveAction"
    assert turn.plan.core_action.direction is Direction.RIGHT
    assert memory.last_intents["cargo_intercepting"] == 1
    assert memory.core_intercept_worker_id == str(U(3))


def test_core_keeps_the_same_cargo_intercept_target_between_steps():
    memory = ScoutMemory(core_intercept_worker_id=str(U(3)))
    turn = make_turn(
        resources=0,
        objects=(
            core_view(position=(1, 0)),
            worker_view((-10, 0), cargo=1, uid=2),
            worker_view((18, 0), cargo=1, uid=3),
            worker_view((30, 0), uid=4),
            worker_view((31, 0), uid=5),
        ),
    )
    decide(turn, memory)
    assert core_action_type(turn) == "StartMoveAction"
    assert turn.plan.core_action.direction is Direction.RIGHT
    assert memory.core_intercept_worker_id == str(U(3))


def attempt_economic_core_move(
    memory: ScoutMemory,
    *,
    tick: int,
    activity_points: list[tuple[int, int]],
    objects: tuple | None = None,
    active_economic_workers: int = 2,
):
    from tactic import MovementReservations, start_economic_core_move

    turn = make_turn(
        resources=0,
        tick=tick,
        objects=economic_migration_objects() if objects is None else objects,
    )
    started = start_economic_core_move(
        turn,
        (),
        TickBudget(resources=0, space=10),
        frozenset(),
        set(),
        MovementReservations(),
        activity_points,
        active_economic_workers,
        False,
        memory,
        Counter(),
    )
    return turn, started


def test_core_keeps_locked_migration_goal_when_activity_flips_behind_it():
    memory = ScoutMemory(
        core_migration_goal=(20, 0),
        core_migration_goal_kind="activity",
    )
    turn, started = attempt_economic_core_move(
        memory,
        tick=105,
        activity_points=[(-20, 0), (-20, 2)],
    )
    assert started is True
    assert turn.plan.core_action.direction is Direction.RIGHT
    assert memory.core_migration_goal == (20, 0)


def test_density_migration_goal_clears_after_reaching_a_rich_chunk():
    core_position = (-400, 0)
    memory = ScoutMemory(
        core_migration_goal=(0, 0),
        core_migration_goal_kind="density",
    )
    objects = (
        core_view(position=core_position),
        *(worker_view((core_position[0] + uid, 0), uid=uid) for uid in range(2, 7)),
    )
    turn, started = attempt_economic_core_move(
        memory,
        tick=105,
        activity_points=[],
        objects=objects,
    )
    assert started is False
    assert core_action_type(turn) is None
    assert memory.core_migration_goal is None
    assert memory.last_migration_hold == "goal_invalidated"


def test_activity_migration_goal_clears_without_active_economic_workers():
    memory = ScoutMemory(
        core_migration_goal=(20, 0),
        core_migration_goal_kind="activity",
    )
    turn, started = attempt_economic_core_move(
        memory,
        tick=105,
        activity_points=[],
        active_economic_workers=0,
    )
    assert started is False
    assert core_action_type(turn) is None
    assert memory.core_migration_goal is None
    assert memory.last_migration_hold == "goal_invalidated"


def test_activity_migration_goal_clears_when_its_chunk_becomes_poorer():
    core_position = (-64, 0)
    memory = ScoutMemory(
        core_migration_goal=(400, 0),
        core_migration_goal_kind="activity",
    )
    objects = (
        core_view(position=core_position),
        *(worker_view((core_position[0] + uid, 0), uid=uid) for uid in range(2, 7)),
    )
    turn, started = attempt_economic_core_move(
        memory,
        tick=105,
        activity_points=[(400, 0), (400, 2)],
        objects=objects,
    )
    assert started is False
    assert core_action_type(turn) is None
    assert memory.core_migration_goal is None
    assert memory.last_migration_hold == "goal_invalidated"


def test_queued_core_move_does_not_start_reverse_cooldown_before_resolution():
    memory = ScoutMemory()
    turn, started = attempt_economic_core_move(
        memory,
        tick=105,
        activity_points=[(20, 0), (20, 2)],
    )
    assert started is True
    assert core_action_type(turn) == "StartMoveAction"
    assert memory.core_last_move_delta is None
    assert memory.core_last_move_tick == -1


def test_failed_or_cancelled_core_move_clears_reverse_cooldown():
    from tactic import observe

    for index, event_type in enumerate(
        ("CORE_MOVE_START_FAILED", "CORE_MOVE_FAILED", "CORE_MOVE_CANCELLED")
    ):
        event = ResolutionEvent(
            event_id=U(80 + index),
            tick=104,
            event_type=event_type,
            reason_code="MOVE_CONTESTED" if event_type.endswith("FAILED") else None,
            actor_id=U(1),
            position=(0, 0),
        )
        memory = ScoutMemory(
            core_last_move_delta=(1, 0),
            core_last_move_tick=100,
        )
        turn = make_turn(events=(event,), objects=(core_view(), worker_view((5, 0))))
        observe(turn, memory)
        assert memory.core_last_move_delta is None
        assert memory.core_last_move_tick == -1


def test_core_blocks_an_ordinary_reverse_during_migration_cooldown():
    memory = ScoutMemory(
        core_migration_goal=(-20, 0),
        core_migration_goal_kind="activity",
        core_last_move_delta=(1, 0),
        core_last_move_tick=100,
    )
    turn, started = attempt_economic_core_move(
        memory,
        tick=111,
        activity_points=[(-20, 0), (-20, 2)],
    )
    assert started is False
    assert core_action_type(turn) is None
    assert memory.last_migration_hold == "reverse_cooldown"


def test_core_may_reverse_after_migration_cooldown():
    memory = ScoutMemory(
        core_migration_goal=(-20, 0),
        core_migration_goal_kind="activity",
        core_last_move_delta=(1, 0),
        core_last_move_tick=100,
    )
    turn, started = attempt_economic_core_move(
        memory,
        tick=112,
        activity_points=[(-20, 0), (-20, 2)],
    )
    assert started is True
    assert turn.plan.core_action.direction is Direction.LEFT
    assert memory.last_migration_hold is None


def test_empty_core_may_reverse_immediately_to_intercept_cargo():
    memory = ScoutMemory(
        core_last_move_delta=(1, 0),
        core_last_move_tick=100,
    )
    objects = (
        core_view(),
        worker_view((-20, 0), cargo=1, uid=2),
        worker_view((10, 0), uid=3),
        worker_view((11, 0), uid=4),
        worker_view((12, 0), uid=5),
    )
    turn, started = attempt_economic_core_move(
        memory,
        tick=101,
        activity_points=[(20, 0), (20, 2)],
        objects=objects,
    )
    assert started is True
    assert turn.plan.core_action.direction is Direction.LEFT
    assert memory.core_intercept_worker_id == str(U(2))


def test_understaffed_empty_core_still_intercepts_surviving_cargo():
    turn = make_turn(
        resources=0,
        objects=(
            core_view(),
            worker_view((20, 0), cargo=1, uid=2),
            worker_view((30, 0), cargo=1, uid=3),
            worker_view((40, 0), cargo=1, uid=4),
        ),
    )
    memory = ScoutMemory()
    decide(turn, memory)
    assert core_action_type(turn) == "StartMoveAction"
    assert memory.last_intents["cargo_intercepting"] == 1
    assert memory.core_intercept_worker_id == str(U(2))


def test_core_waits_when_cargo_will_arrive_during_migration():
    memory = ScoutMemory(
        known_resources={(20, 0)},
        resource_last_seen={(20, 0): 100},
    )
    turn = make_turn(
        resources=0,
        objects=economic_migration_objects(cargo_distance=4),
    )
    decide(turn, memory)
    assert core_action_type(turn) == "WaitAction"


def test_core_does_not_migrate_when_nearly_ready_to_build_the_guard():
    memory = ScoutMemory(
        known_resources={(20, 0)},
        resource_last_seen={(20, 0): 100},
    )
    turn = make_turn(
        resources=9,
        objects=economic_migration_objects(),
    )
    decide(turn, memory)
    assert core_action_type(turn) == "WaitAction"


def test_worker_rendezvous_with_moving_core_instead_of_depositing():
    moving_core = core_view(
        state=CoreState.MOVING,
        move_direction=Direction.RIGHT,
        move_progress=2,
        move_required_ticks=4,
        destination=(1, 0),
    )
    turn = make_turn(
        resources=0,
        objects=(moving_core, worker_view((0, 0), cargo=1, uid=2)),
    )
    decide(turn)
    assert action_type(turn.plan, 2) == "MoveAction"
    assert direction_of(turn.plan, 2) is Direction.RIGHT
    assert core_action_type(turn) == "WaitAction"


def test_only_one_unit_waits_on_the_moving_core_destination():
    moving_core = core_view(
        state=CoreState.MOVING,
        move_direction=Direction.RIGHT,
        move_progress=2,
        move_required_ticks=4,
        destination=(1, 0),
    )
    turn = make_turn(
        resources=0,
        objects=(
            moving_core,
            worker_view((1, 0), cargo=1, uid=2),
            worker_view((1, 0), cargo=1, uid=3),
        ),
    )
    decide(turn)
    actions = [action_type(turn.plan, 2), action_type(turn.plan, 3)]
    assert actions.count("WaitAction") == 1
    assert actions.count("MoveAction") == 1


def test_moving_core_cancels_if_its_destination_becomes_a_resource():
    moving_core = core_view(
        state=CoreState.MOVING,
        move_direction=Direction.RIGHT,
        move_progress=2,
        move_required_ticks=4,
        destination=(1, 0),
    )
    turn = make_turn(
        resources=0,
        objects=(
            moving_core,
            worker_view((10, 0), uid=2),
            terrain("RESOURCE", [(1, 0)]),
        ),
    )
    decide(turn)
    assert core_action_type(turn) == "CancelMoveAction"


def test_moving_core_cancels_if_destination_enters_ranger_fire():
    moving_core = core_view(
        state=CoreState.MOVING,
        move_direction=Direction.RIGHT,
        move_progress=2,
        move_required_ticks=4,
        destination=(1, 0),
    )
    turn = make_turn(
        resources=0,
        objects=(
            moving_core,
            worker_view((10, 0), uid=2),
            enemy_view((4, 0), unit_type=UnitType.RANGER),
        ),
    )
    decide(turn)
    assert core_action_type(turn) == "CancelMoveAction"


def test_moving_core_cancels_when_current_and_destination_share_ranger_fire():
    moving_core = core_view(
        state=CoreState.MOVING,
        move_direction=Direction.RIGHT,
        move_progress=1,
        move_required_ticks=4,
        destination=(1, 0),
    )
    turn = make_turn(
        resources=20,
        objects=(
            moving_core,
            worker_view((10, 0), uid=2),
            enemy_view((3, 0), unit_type=UnitType.RANGER),
        ),
    )
    decide(turn)
    assert core_action_type(turn) == "CancelMoveAction"


def test_remote_low_quota_core_relocates_toward_richer_chunks():
    turn = make_turn(
        resources=0,
        objects=(
            core_view(position=(-800, 0)),
            worker_view((-790, 0), uid=2),
            worker_view((-790, 1), uid=3),
            worker_view((-790, 2), uid=4),
            worker_view((-790, 3), uid=5),
        ),
    )
    memory = ScoutMemory()
    decide(turn, memory)
    assert core_action_type(turn) == "StartMoveAction"
    assert turn.plan.core_action.direction is Direction.RIGHT
    assert memory.last_intents["density_relocating"] == 1
    assert memory.core_migration_goal == (-768, 0)
    assert memory.core_migration_goal_kind == "density"


def test_density_migration_milestone_is_bounded_to_one_chunk():
    from tactic import core_migration_milestone

    assert core_migration_milestone((-800, 0), (0, 0)) == (-768, 0)
    assert core_migration_milestone((-10, -10), (0, 0)) == (0, 0)
    assert core_migration_milestone((-20, -20), (20, 20)) == (12, -20)


def test_legacy_long_density_goal_is_trimmed_before_migration_resumes():
    memory = ScoutMemory(
        core_migration_goal=(0, 0),
        core_migration_goal_kind="density",
    )
    turn = make_turn(
        resources=0,
        objects=(
            core_view(position=(-800, 0)),
            *(worker_view((-790, uid), uid=uid) for uid in range(2, 6)),
        ),
    )
    decide(turn, memory)
    assert memory.core_migration_goal == (-768, 0)
    assert core_action_type(turn) == "StartMoveAction"


def test_legacy_density_goal_is_trimmed_even_while_waiting_for_cargo():
    memory = ScoutMemory(
        core_migration_goal=(0, 0),
        core_migration_goal_kind="density",
    )
    turn = make_turn(
        resources=0,
        objects=(
            core_view(position=(-800, 0)),
            worker_view((-790, 0), cargo=1, uid=2),
            *(worker_view((-790, uid), uid=uid) for uid in range(3, 7)),
        ),
    )
    decide(turn, memory)
    assert memory.core_migration_goal == (-768, 0)


def test_reaching_density_milestone_forces_a_fresh_economic_evaluation():
    memory = ScoutMemory(
        core_migration_goal=(-768, 0),
        core_migration_goal_kind="density",
    )
    turn = make_turn(
        resources=0,
        objects=(
            core_view(position=(-775, 0)),
            *(worker_view((-765, uid), uid=uid) for uid in range(2, 6)),
        ),
    )
    decide(turn, memory)
    assert memory.core_migration_goal is None
    assert memory.last_migration_hold == "goal_reached"
    assert core_action_type(turn) == "WaitAction"


def test_recent_core_damage_pauses_economic_migration():
    memory = ScoutMemory(
        known_resources={(20, 0), (20, 2)},
        resource_last_seen={(20, 0): 100, (20, 2): 100},
    )
    event = ResolutionEvent(
        event_id=U(80),
        tick=99,
        event_type="CORE_DAMAGED",
        reason_code="ATTACK",
        target_id=U(1),
        position=(0, 0),
        values={"damage": 1, "shield_damage": 1, "hp_damage": 0},
    )
    turn = make_turn(
        resources=0,
        events=(event,),
        objects=economic_migration_objects(),
    )
    decide(turn, memory)
    assert core_action_type(turn) == "WaitAction"
    assert memory.core_threat_until_tick > turn.tick


def fresh_route_memory() -> ScoutMemory:
    return ScoutMemory(
        known_resources={(20, 0), (20, 2)},
        resource_last_seen={(20, 0): 100, (20, 2): 100},
    )


def test_harmless_enemy_worker_near_the_core_does_not_freeze_migration():
    memory = fresh_route_memory()
    turn = make_turn(
        resources=0,
        objects=(
            *economic_migration_objects(),
            enemy_view((0, 5), uid=90, unit_type=UnitType.WORKER),
        ),
    )
    decide(turn, memory)
    assert core_action_type(turn) == "StartMoveAction"
    assert turn.plan.core_action.direction is Direction.RIGHT
    assert memory.last_migration_hold is None


def test_enemy_core_near_our_core_does_not_freeze_migration():
    memory = fresh_route_memory()
    turn = make_turn(
        resources=0,
        objects=(
            *economic_migration_objects(),
            enemy_view((0, 5), uid=91, kind="CORE"),
        ),
    )
    decide(turn, memory)
    assert core_action_type(turn) == "StartMoveAction"
    assert memory.last_migration_hold is None


def test_combat_enemy_inside_the_buffer_still_holds_migration():
    from tactic import MovementReservations, start_economic_core_move

    def attempt(enemy) -> tuple[bool, ScoutMemory]:
        turn = make_turn(resources=0, objects=economic_migration_objects())
        memory = ScoutMemory()
        started = start_economic_core_move(
            turn,
            (enemy,),
            TickBudget(resources=0, space=10),
            frozenset(),
            set(),
            MovementReservations(),
            [(20, 0), (20, 2)],
            2,
            False,
            memory,
            Counter(),
        )
        return started, memory

    started, memory = attempt(enemy_view((0, 5), uid=90, unit_type=UnitType.RANGER))
    assert started is False
    assert memory.last_migration_hold == "combat_enemy_near_core"

    started, memory = attempt(enemy_view((0, 5), uid=90, unit_type=UnitType.WORKER))
    assert started is True
    assert memory.last_migration_hold is None


def test_migration_hold_reason_explains_a_waiting_core():
    memory = fresh_route_memory()
    turn = make_turn(
        resources=0,
        objects=economic_migration_objects(cargo_distance=4),
    )
    decide(turn, memory)
    assert core_action_type(turn) == "WaitAction"
    assert memory.last_migration_hold == "cargo_already_arriving"


def test_core_summary_reports_harmless_enemies_separately():
    from tactic import summarize_core_state

    turn = make_turn(
        objects=(
            core_view(),
            worker_view((3, 0), uid=2),
            enemy_view((0, 5), uid=90, unit_type=UnitType.WORKER),
        ),
    )
    summary = summarize_core_state(turn, frozenset())
    assert "enemy:-" in summary
    assert "other:5" in summary


# Chunk (-26, -8) sits 32 rings out, where the documented node quota is 3.
BARREN_CORE = (-813, -250)


def drought_history(tick: int = 100) -> list[tuple[int, int, int, int]]:
    from tactic import STARVATION_SAMPLES

    return [(sample, 0, 0, 0) for sample in range(tick - STARVATION_SAMPLES + 1, tick + 1)]


def test_starving_core_walks_out_of_a_barren_chunk_instead_of_chasing_cargo():
    memory = ScoutMemory(economic_history=drought_history())
    turn = make_turn(
        resources=0,
        tick=100,
        objects=(
            core_view(position=BARREN_CORE),
            worker_view((BARREN_CORE[0] - 40, BARREN_CORE[1]), cargo=1, uid=2),
        ),
    )
    decide(turn, memory)
    assert core_action_type(turn) == "StartMoveAction"
    # RIGHT is toward the origin; the carrier sits 40 cells to the LEFT.
    assert turn.plan.core_action.direction is Direction.RIGHT
    assert memory.last_intents["desert_escaping"] == 1
    assert memory.core_intercept_worker_id is None


def test_starving_core_escapes_even_with_a_skeleton_fleet():
    memory = ScoutMemory(economic_history=drought_history())
    turn = make_turn(
        resources=0,
        tick=100,
        objects=(
            core_view(position=BARREN_CORE),
            worker_view((BARREN_CORE[0] - 7, BARREN_CORE[1]), uid=2),
        ),
    )
    decide(turn, memory)
    assert core_action_type(turn) == "StartMoveAction"
    assert turn.plan.core_action.direction is Direction.RIGHT
    assert memory.last_intents["desert_escaping"] == 1


def test_starving_core_in_a_rich_chunk_stays_put():
    memory = ScoutMemory(economic_history=drought_history())
    turn = make_turn(
        resources=0,
        tick=100,
        objects=(core_view(), worker_view((7, 0), uid=2)),
    )
    decide(turn, memory)
    assert core_action_type(turn) == "WaitAction"
    assert memory.last_intents["desert_escaping"] == 0
    assert memory.last_migration_hold == "fleet_too_small"


def test_a_remembered_node_keeps_the_core_from_abandoning_the_chunk():
    node = (BARREN_CORE[0] + 13, BARREN_CORE[1])
    memory = ScoutMemory(
        economic_history=drought_history(),
        known_resources={node},
        resource_last_seen={node: 100},
    )
    turn = make_turn(
        resources=0,
        tick=100,
        objects=(
            core_view(position=BARREN_CORE),
            worker_view((BARREN_CORE[0] - 40, BARREN_CORE[1]), cargo=1, uid=2),
        ),
    )
    decide(turn, memory)
    assert core_action_type(turn) == "StartMoveAction"
    # Still intercepting the carrier on the left rather than leaving for origin.
    assert turn.plan.core_action.direction is Direction.LEFT
    assert memory.last_intents["desert_escaping"] == 0


def test_core_routes_around_a_resource_cell():
    memory = ScoutMemory(
        known_resources={(1, 0), (20, 0)},
        resource_last_seen={(1, 0): 100, (20, 0): 100},
    )
    turn = make_turn(
        resources=0,
        objects=(
            *economic_migration_objects(),
            terrain("RESOURCE", [(1, 0)]),
        ),
    )
    decide(turn, memory)
    assert core_action_type(turn) == "StartMoveAction"
    assert turn.plan.core_action.direction is not Direction.RIGHT


def test_nearest_deposit_eta_accounts_for_core_migration():
    from tactic import nearest_deposit_eta

    moving_core = core_view(
        state=CoreState.MOVING,
        move_direction=Direction.RIGHT,
        move_progress=2,
        move_required_ticks=4,
        destination=(1, 0),
    )
    turn = make_turn(
        resources=0,
        objects=(moving_core, worker_view((0, 0), cargo=1, uid=2)),
    )
    assert nearest_deposit_eta(turn) == 2


def test_intent_metrics_cover_worker_and_guard_roles():
    memory = ScoutMemory()
    turn = make_turn(
        resources=0,
        objects=(
            core_view(),
            worker_view((0, 0), cargo=1, uid=2),
            worker_view((4, 0), uid=3),
            vanguard_view((0, 3), uid=4),
        ),
    )
    decide(turn, memory)
    assert memory.last_intents["depositing"] == 1
    assert memory.last_intents["scouting"] == 1
    assert memory.last_intents["guarding"] == 1


def test_fleet_summary_reports_each_unit_type():
    from tactic import summarize_fleet

    turn = make_turn(
        objects=(
            core_view(),
            worker_view((1, 0), uid=2),
            worker_view((2, 0), uid=3),
            vanguard_view((3, 0), uid=4),
            ranger_view((4, 0), uid=5),
        ),
    )
    assert summarize_fleet(turn) == "fleet[w:2,v:1,r:1]"


def test_lone_guard_holds_when_only_enemy_is_far_from_core():
    memory = ScoutMemory()
    turn = make_turn(
        resources=0,
        objects=(
            core_view(),
            worker_view((4, 0), uid=2),
            vanguard_view((0, 2), uid=4),
            enemy_view((20, 0)),
        ),
    )
    decide(turn, memory)
    assert memory.last_intents["guarding"] == 1
    assert memory.last_intents["engaging"] == 0


def test_vanguard_sweeps_adjacent_enemy_even_when_far_from_core():
    """Local contact takes priority over the first-unit guard assignment."""

    memory = ScoutMemory()
    turn = make_turn(
        resources=0,
        objects=(
            core_view(),
            vanguard_view((20, 0), uid=4),
            enemy_view((21, 0), uid=90),
        ),
    )
    decide(turn, memory)
    assert action_type(turn.plan, 4) == "SweepAction"
    assert direction_of(turn.plan, 4) is Direction.RIGHT
    assert memory.last_intents["engaging"] == 1
    assert memory.last_intents["guarding"] == 0


def test_distant_ordinary_enemy_does_not_draw_the_combat_fleet():
    memory = ScoutMemory()
    defenders = tuple(
        vanguard_view((0, index + 2), uid=uid)
        for index, uid in enumerate(range(10, 17))
    )
    turn = make_turn(
        resources=0,
        objects=(core_view(), worker_view((2, 0), uid=2), *defenders, enemy_view((60, 0))),
    )
    decide(turn, memory)
    assert memory.last_intents["engaging"] == 0
    assert memory.last_intents["limited_pursuit"] == 0


def test_ordinary_enemy_inside_the_leash_draws_only_two_hunters():
    memory = ScoutMemory()
    defenders = tuple(
        vanguard_view((0, index + 2), uid=uid)
        for index, uid in enumerate(range(10, 17))
    )
    turn = make_turn(
        resources=0,
        objects=(core_view(), worker_view((2, 0), uid=2), *defenders, enemy_view((12, 0))),
    )
    decide(turn, memory)
    assert memory.last_intents["engaging"] == 2
    assert memory.last_intents["limited_pursuit"] == 2


def test_immediate_core_threat_draws_the_full_combat_fleet():
    memory = ScoutMemory()
    defenders = tuple(
        vanguard_view((0, index + 2), uid=uid)
        for index, uid in enumerate(range(10, 17))
    )
    turn = make_turn(
        resources=0,
        objects=(core_view(), worker_view((2, 0), uid=2), *defenders, enemy_view((1, 0))),
    )
    decide(turn, memory)
    assert memory.last_intents["engaging"] == len(defenders)


def test_distant_beacon_carrier_draws_only_two_hunters():
    memory = ScoutMemory()
    defenders = tuple(
        ranger_view((0, index + 2), uid=uid)
        for index, uid in enumerate(range(10, 17))
    )
    carrier = enemy_view((60, 0), uid=90)
    turn = make_turn(
        resources=0,
        objects=(core_view(), worker_view((2, 0), uid=2), *defenders, carrier),
        beacon=ChampionBeacon(
            position=carrier.position,
            status=BeaconStatus.CARRIED,
            carrier_id=carrier.id,
        ),
    )
    decide(turn, memory)
    assert memory.last_intents["engaging"] == 2
    assert memory.last_intents["carrier_hunting"] == 2


# --- bounded scouting and reachable-resource scheduling ----------------------


def test_scout_grid_keys_do_not_move_with_the_core():
    from tactic import scout_grid_center, scout_grid_key

    assert scout_grid_key((12, 4), (0, 0)) == (4, 1)
    assert scout_grid_key((12, 4), (8, 0)) == (4, 1)
    assert scout_grid_center((4, 1), (0, 0)) == (13, 4)
    assert scout_grid_center((4, 1), (8, 0)) == (13, 4)


def test_chunk_resource_quota_matches_central_and_remote_rings():
    from tactic import chunk_resource_quota

    assert chunk_resource_quota((0, 0)) == 16
    assert chunk_resource_quota((-1, -1)) == 16
    assert chunk_resource_quota((32, 0)) == 14
    assert chunk_resource_quota((100_000, 0)) == 2


def test_absolute_scout_frontier_stays_within_the_core_tether():
    from tactic import SCOUT_MAX_DISTANCE, scout_coverage_target

    memory = ScoutMemory(
        scout_seen={(gx, gy): 90 for gx in range(-12, 13) for gy in range(-12, 13)}
    )
    turn = make_turn(objects=(core_view(), worker_view((0, 0), uid=2)))
    target = scout_coverage_target(turn.workers[0], (0, 0), memory, set(), turn.tick)
    assert target is not None
    assert abs(target[0]) + abs(target[1]) <= SCOUT_MAX_DISTANCE


def test_resource_assignment_rejects_an_uneconomic_far_route():
    from tactic import assign_resource_targets

    turn = make_turn(objects=(core_view(), worker_view((0, 0), uid=2)))
    far = (100, 0)
    assert assign_resource_targets(turn.workers, {far}) == {}


def test_stranded_worker_is_not_sent_to_another_distant_resource_cluster():
    from tactic import assign_resource_targets

    turn = make_turn(objects=(core_view(), worker_view((100, 0), uid=2)))
    assert assign_resource_targets(
        turn.workers,
        {(200, 0)},
        depot=(0, 0),
    ) == {}


def test_resource_assignment_rejects_a_long_return_route():
    from tactic import assign_resource_targets

    resource = (100, 0)
    turn = make_turn(
        objects=(core_view(), worker_view((resource[0] - 1, 0), uid=2))
    )
    assert assign_resource_targets(turn.workers, {resource}, depot=(0, 0)) == {}


def test_resource_assignment_caps_the_complete_round_trip():
    from tactic import assign_resource_targets

    turn = make_turn(objects=(core_view(), worker_view((0, 0), uid=2)))
    assert assign_resource_targets(
        turn.workers,
        {(40, 0)},
        depot=(0, 0),
        max_total_cost=64,
    ) == {}
    assert assign_resource_targets(
        turn.workers,
        {(30, 0)},
        depot=(0, 0),
        max_total_cost=64,
    ) == {U(2): (30, 0)}


def test_resource_trip_budget_tightens_during_a_delivery_drought():
    from tactic import RESOURCE_TRIP_COST_RECOVERY, resource_round_trip_budget

    memory = ScoutMemory(
        economic_history=[(tick, 0, 0, 0) for tick in range(85, 101)]
    )
    turn = make_turn(
        tick=100,
        resources=0,
        objects=(core_view(), worker_view((0, 0), uid=2)),
    )
    assert resource_round_trip_budget(turn, memory) == RESOURCE_TRIP_COST_RECOVERY


def test_resource_trip_budget_expands_only_after_healthy_income():
    from tactic import RESOURCE_TRIP_COST_HEALTHY, resource_round_trip_budget

    memory = ScoutMemory(economic_history=[(100, 2, 2, 0)])
    turn = make_turn(
        tick=100,
        resources=10,
        objects=(
            core_view(),
            worker_view((0, 0), uid=2),
            worker_view((1, 0), uid=3),
            worker_view((2, 0), uid=4),
            worker_view((3, 0), uid=5),
        ),
    )
    assert resource_round_trip_budget(turn, memory) == RESOURCE_TRIP_COST_HEALTHY


def test_drought_stretches_one_trip_when_the_tight_budget_refuses_every_node():
    from tactic import RESOURCE_TRIP_COST_HEALTHY

    memory = ScoutMemory(economic_history=drought_history())
    turn = make_turn(
        resources=0,
        tick=100,
        objects=(
            core_view(),
            worker_view((1, 0), uid=2),
            terrain("RESOURCE", [(28, 0)]),
        ),
    )
    decide(turn, memory)
    # 27 out plus 28 home is refused by the 48-cost drought budget.
    assert memory.resource_assignments == {str(U(2)): (28, 0)}
    assert memory.last_intents["stretching_trip"] == 1
    assert memory.last_trip_budget == RESOURCE_TRIP_COST_HEALTHY


def test_drought_does_not_stretch_past_the_healthy_budget():
    from tactic import RESOURCE_TRIP_COST_RECOVERY

    memory = ScoutMemory(economic_history=drought_history())
    turn = make_turn(
        resources=0,
        tick=100,
        objects=(
            core_view(),
            worker_view((1, 0), uid=2),
            terrain("RESOURCE", [(40, 0)]),
        ),
    )
    decide(turn, memory)
    assert memory.resource_assignments == {}
    assert memory.last_intents["stretching_trip"] == 0
    assert memory.last_trip_budget == RESOURCE_TRIP_COST_RECOVERY


def test_scout_disc_widens_only_after_the_near_grid_is_explored():
    from tactic import (
        SCOUT_FAR_DISTANCE,
        SCOUT_MAX_DISTANCE,
        scout_disc_radius,
        scout_grid_disc,
    )

    memory = ScoutMemory()
    assert scout_disc_radius((0, 0), memory) == SCOUT_MAX_DISTANCE

    disc = scout_grid_disc((0, 0), SCOUT_MAX_DISTANCE)
    for cell in disc[: len(disc) // 2]:
        memory.scout_seen[cell] = 100
    assert scout_disc_radius((0, 0), memory) == SCOUT_MAX_DISTANCE

    for cell in disc:
        memory.scout_seen[cell] = 100
    assert scout_disc_radius((0, 0), memory) == SCOUT_FAR_DISTANCE


def test_exhausted_scout_disc_sends_a_worker_past_the_near_radius():
    from tactic import (
        SCOUT_MAX_DISTANCE,
        scout_coverage_target,
        scout_disc_radius,
        scout_grid_disc,
    )

    memory = ScoutMemory()
    for cell in scout_grid_disc((0, 0), SCOUT_MAX_DISTANCE):
        memory.scout_seen[cell] = 100
    turn = make_turn(objects=(core_view(), worker_view((0, 1), uid=2)), tick=100)
    target = scout_coverage_target(
        turn.workers[0],
        (0, 0),
        memory,
        set(),
        100,
        frozenset(),
        scout_disc_radius((0, 0), memory),
    )
    assert target is not None
    assert abs(target[0]) + abs(target[1]) > SCOUT_MAX_DISTANCE


def test_visible_resource_assignment_overrides_a_remote_scout_role():
    from tactic import ensure_scout_roles

    workers = tuple(worker_view((uid, 0), uid=uid) for uid in range(2, 16))
    memory = ScoutMemory()
    ensure_scout_roles({str(worker.id) for worker in workers}, memory)
    remote_id = min(
        worker_id for worker_id, role in memory.scout_roles.items() if role == "remote"
    )
    remote_worker = next(worker for worker in workers if str(worker.id) == remote_id)
    turn = make_turn(
        resources=0,
        objects=(core_view(), *workers, terrain("RESOURCE", [remote_worker.position])),
    )
    decide(turn, memory)
    assert action_type(turn.plan, int(remote_id[-2:], 16)) == "HarvestAction"


def test_distant_visible_resource_under_worker_is_harvested_immediately():
    turn = make_turn(
        resources=0,
        objects=(
            core_view(),
            worker_view((100, 0), uid=2),
            terrain("RESOURCE", [(100, 0)]),
        ),
    )
    decide(turn)
    assert action_type(turn.plan, 2) == "HarvestAction"


def test_underfoot_resource_beats_an_explicit_helper_distance_limit():
    from tactic import assign_resource_targets

    turn = make_turn(objects=(core_view(), worker_view((100, 0), uid=2)))
    assert assign_resource_targets(
        turn.workers,
        {(100, 0)},
        max_cost=20,
        depot=(0, 0),
    ) == {U(2): (100, 0)}


def test_live_style_far_resource_is_not_assigned_for_a_long_return():
    from tactic import assign_resource_targets

    resource = (-664, -223)
    turn = make_turn(
        objects=(
            core_view(position=(-855, -255)),
            worker_view((-1675, -272), uid=2),
            worker_view((-402, -267), uid=3),
            worker_view((-842, -992), uid=4),
            worker_view((-614, -221), uid=5),
        ),
    )
    assert assign_resource_targets(
        turn.workers,
        {resource},
        depot=(-855, -255),
    ) == {}


def test_remote_resource_assignments_are_limited_per_fleet():
    from tactic import assign_resource_targets

    turn = make_turn(
        objects=(
            core_view(),
            worker_view((0, 0), uid=2),
            worker_view((1, 0), uid=3),
            worker_view((2, 0), uid=4),
            worker_view((3, 0), uid=5),
        )
    )
    targets = assign_resource_targets(
        turn.workers,
        {(30, 0), (31, 0), (32, 0), (33, 0)},
        depot=(0, 0),
        remote_distance=24,
        max_remote_workers=1,
    )
    assert len(targets) == 1


def test_six_safe_workers_use_two_reachable_remote_resource_slots():
    workers = tuple(
        worker_view((uid - 1, 0), uid=uid)
        for uid in range(2, 8)
    )
    memory = ScoutMemory()
    turn = make_turn(
        resources=0,
        objects=(
            core_view(),
            *workers,
            terrain("RESOURCE", [(25, 0), (25, 1)]),
        ),
    )
    decide(turn, memory)
    assert len(memory.resource_assignments) == 2
    assert memory.last_intents["mining"] == 2


def test_five_workers_keep_one_remote_resource_slot():
    workers = tuple(
        worker_view((uid - 1, 0), uid=uid)
        for uid in range(2, 7)
    )
    memory = ScoutMemory()
    turn = make_turn(
        resources=0,
        objects=(
            core_view(),
            *workers,
            terrain("RESOURCE", [(25, 0), (25, 1)]),
        ),
    )
    decide(turn, memory)
    assert len(memory.resource_assignments) == 1
    assert memory.last_intents["mining"] == 1


def test_visible_combat_enemy_keeps_one_remote_resource_slot():
    workers = tuple(
        worker_view((uid - 1, 0), uid=uid)
        for uid in range(2, 8)
    )
    memory = ScoutMemory()
    turn = make_turn(
        resources=0,
        objects=(
            core_view(),
            *workers,
            terrain("RESOURCE", [(25, 0), (25, 1)]),
            enemy_view((0, 5)),
        ),
    )
    decide(turn, memory)
    assert len(memory.resource_assignments) == 1
    assert memory.last_intents["mining"] == 1


def test_recent_defense_caution_keeps_one_remote_resource_slot():
    workers = tuple(
        worker_view((uid - 1, 0), uid=uid)
        for uid in range(2, 8)
    )
    memory = ScoutMemory(core_threat_until_tick=100)
    turn = make_turn(
        resources=0,
        tick=100,
        objects=(
            core_view(),
            *workers,
            terrain("RESOURCE", [(25, 0), (25, 1)]),
        ),
    )
    decide(turn, memory)
    assert len(memory.resource_assignments) == 1
    assert memory.last_intents["mining"] == 1


def test_returning_remote_worker_uses_one_of_two_adaptive_slots():
    memory = ScoutMemory()
    turn = make_turn(
        resources=0,
        objects=(
            core_view(),
            worker_view((30, 0), cargo=1, uid=2),
            *(worker_view((uid - 2, 0), uid=uid) for uid in range(3, 8)),
            terrain("RESOURCE", [(25, 0), (25, 1)]),
        ),
    )
    decide(turn, memory)
    assert len(memory.resource_assignments) == 1
    assert memory.last_intents["mining"] == 1


def test_returning_remote_worker_consumes_the_fleet_expedition_slot():
    turn = make_turn(
        resources=0,
        objects=(
            core_view(),
            worker_view((30, 0), cargo=1, uid=2),
            worker_view((1, 0), uid=3),
            worker_view((2, 0), uid=4),
            worker_view((3, 0), uid=5),
            terrain("RESOURCE", [(30, 1)]),
        ),
    )
    memory = ScoutMemory()
    decide(turn, memory)
    assert memory.resource_assignments == {}
    assert memory.last_intents["mining"] == 0


def test_persisted_giant_scout_target_is_replaced_near_the_core():
    from tactic import SCOUT_MAX_DISTANCE, scout_coverage_target, scout_grid_center

    worker = worker_view((-614, -221), uid=5)
    worker_id = str(worker.id)
    giant_cell = (325, -28)
    memory = ScoutMemory(scout_targets={worker_id: giant_cell})
    target = scout_coverage_target(
        worker,
        (-855, -255),
        memory,
        set(),
        123400,
    )
    assert target is not None
    assert memory.scout_targets[worker_id] != giant_cell
    assert abs(target[0] + 855) + abs(target[1] + 255) <= SCOUT_MAX_DISTANCE
    assert scout_grid_center(giant_cell) != target


def test_remote_unassigned_worker_is_recalled_instead_of_scouting_farther():
    memory = ScoutMemory()
    turn = make_turn(
        resources=0,
        objects=(core_view(), worker_view((100, 0), uid=2)),
    )
    decide(turn, memory)
    assert action_type(turn.plan, 2) == "MoveAction"
    assert direction_of(turn.plan, 2) is Direction.LEFT
    assert memory.last_intents["recalling"] == 1
    assert memory.last_intents["scouting"] == 0


def test_new_visible_resource_interrupts_an_existing_scout_route():
    worker_id = str(U(2))
    memory = ScoutMemory(scout_targets={worker_id: (3, 0)})
    turn = make_turn(
        resources=0,
        objects=(
            core_view(),
            worker_view((5, 0), uid=2),
            terrain("RESOURCE", [(4, 0)]),
        ),
    )
    decide(turn, memory)
    assert direction_of(turn.plan, 2) is Direction.LEFT
    assert memory.resource_assignments[worker_id] == (4, 0)
    assert worker_id not in memory.scout_targets


def test_stalled_resource_route_is_released_and_cooled_down():
    from tactic import ResourceProgress

    worker_id = str(U(2))
    target = (10, 0)
    memory = ScoutMemory(
        known_resources={target},
        resource_last_seen={target: 100},
        resource_assignments={worker_id: target},
        resource_progress={
            worker_id: ResourceProgress(target, best_cost=10, stalled_ticks=5)
        },
    )
    turn = make_turn(
        tick=101,
        resources=0,
        objects=(core_view(), worker_view((0, 0), uid=2)),
    )
    decide(turn, memory)
    assert worker_id not in memory.resource_assignments
    assert memory.resource_cooldowns[(worker_id, target)] > turn.tick


def test_worker_route_avoids_visible_enemy_attack_cells():
    turn = make_turn(
        resources=0,
        objects=(
            core_view(),
            worker_view((0, 0), uid=2),
            enemy_view((2, 0), uid=90),
            terrain("RESOURCE", [(4, 0)]),
        ),
    )
    decide(turn)
    assert action_type(turn.plan, 2) == "MoveAction"
    assert direction_of(turn.plan, 2) is not Direction.RIGHT


def test_threat_memory_outlives_lost_vision_then_expires():
    from tactic import THREAT_MEMORY_TICKS, remember_threat_cells

    # A one-Tick memory would still let a carrier walk straight back into the arc.
    assert THREAT_MEMORY_TICKS >= 2

    memory = ScoutMemory()
    seen = make_turn(
        objects=(core_view(), worker_view((5, 5), uid=2), enemy_view((3, 3), uid=90)),
        tick=100,
    )
    arcs = remember_threat_cells(seen, memory, frozenset())
    assert arcs == {(2, 3), (4, 3), (3, 2), (3, 4)}

    blind = make_turn(
        objects=(core_view(), worker_view((5, 5), uid=2)),
        tick=101,
    )
    assert remember_threat_cells(blind, memory, frozenset()) == arcs

    memory.expire(101)
    assert remember_threat_cells(blind, memory, frozenset()) == arcs

    memory.expire(100 + THREAT_MEMORY_TICKS)
    assert remember_threat_cells(blind, memory, frozenset()) == set()


# The detour is only worth taking if it survives the Tick where the shooter
# steps out of vision; otherwise the carrier walks back into the arc and
# oscillates between the same two cells until the game ends.
FLICKER_WALL = [(7, 1), (7, -1), (7, -2), (7, -3)]


def flicker_objects(worker_position, *, enemy_visible: bool) -> tuple:
    objects = [
        core_view(),
        worker_view(worker_position, cargo=1, uid=2),
        terrain("OBSTACLE", FLICKER_WALL),
    ]
    if enemy_visible:
        objects.append(enemy_view((6, 0), uid=90))
    return tuple(objects)


def test_cargo_worker_keeps_its_detour_after_the_threat_leaves_vision():
    memory = ScoutMemory()
    seen = make_turn(
        resources=0,
        objects=flicker_objects((8, 0), enemy_visible=True),
        tick=100,
    )
    decide(seen, memory)
    assert direction_of(seen.plan, 2) is Direction.DOWN

    blind = make_turn(
        resources=0,
        objects=flicker_objects((8, 1), enemy_visible=False),
        tick=101,
    )
    decide(blind, memory)
    assert (7, 0) in memory.threatened
    # Direction.UP would be a step back onto the cell it just left.
    assert direction_of(blind.plan, 2) is Direction.DOWN


def test_looping_cargo_worker_ignores_threat_arcs_to_get_home():
    memory = ScoutMemory(
        scout_positions={str(U(2)): [(2, 0), (3, 0), (2, 0), (3, 0), (2, 0), (3, 0)]}
    )
    turn = make_turn(
        resources=0,
        objects=(
            core_view(),
            worker_view((2, 0), cargo=1, uid=2),
            enemy_view((1, 1), uid=90),
        ),
    )
    decide(turn, memory)
    assert direction_of(turn.plan, 2) is Direction.LEFT
    assert memory.last_intents["loop_breaking"] == 1


def test_cargo_worker_without_a_loop_still_respects_threat_arcs():
    memory = ScoutMemory()
    turn = make_turn(
        resources=0,
        objects=(
            core_view(),
            worker_view((2, 0), cargo=1, uid=2),
            enemy_view((1, 1), uid=90),
        ),
    )
    decide(turn, memory)
    assert direction_of(turn.plan, 2) is not Direction.LEFT
    assert memory.last_intents["loop_breaking"] == 0


def test_threatened_cargo_moves_when_every_neighbor_is_marked_unsafe():
    from tactic import MovementReservations, move_or_escape

    turn = make_turn(
        objects=(core_view(position=(-2, 0)), worker_view((0, 0), cargo=1, uid=2))
    )
    worker = turn.workers[0]
    unsafe = {(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)}
    assert move_or_escape(worker, (-2, 0), unsafe, set(), MovementReservations())
    assert direction_of(turn.plan, 2) is Direction.LEFT


def test_state_loader_preserves_good_fields_when_another_field_is_bad(tmp_path):
    path = tmp_path / "scout_state.json"
    path.write_text(
        """{
          "schema_version": 2,
          "offsets": "broken",
          "known_obstacles": [[4, 5], ["bad"]],
          "known_resources": [[8, 9]],
          "scout_seen": {"1,2": 99, "broken": 100}
        }"""
    )
    memory = ScoutMemory(path=path)
    memory.load()
    assert memory.offsets == {}
    assert memory.known_obstacles == {(4, 5)}
    assert memory.known_resources == {(8, 9)}
    assert memory.scout_seen == {}
    assert memory.dirty is True


def test_state_save_is_throttled_between_checkpoint_ticks(tmp_path):
    path = tmp_path / "scout_state.json"
    memory = ScoutMemory(path=path, known_obstacles={(1, 1)}, dirty=True)
    memory.save(100)

    memory.known_obstacles.add((2, 2))
    memory.dirty = True
    memory.save(101)
    restored = ScoutMemory(path=path)
    restored.load()
    assert restored.known_obstacles == {(1, 1)}

    memory.save(108)
    restored = ScoutMemory(path=path)
    restored.load()
    assert restored.known_obstacles == {(1, 1), (2, 2)}


def test_economic_history_survives_state_restart(tmp_path):
    path = tmp_path / "scout_state.json"
    memory = ScoutMemory(path=path)
    memory.record_economy(100, Counter(harvest=2, deposit=1, dropped=1))
    memory.save(100)

    restored = ScoutMemory(path=path)
    restored.load()
    assert restored.economic_history == [(100, 2, 1, 1)]
    assert restored.economic_totals(100) == Counter(
        harvest=2,
        income=1,
        lost=1,
        samples=1,
    )


def test_core_migration_state_survives_state_restart(tmp_path):
    path = tmp_path / "scout_state.json"
    saved = ScoutMemory(
        path=path,
        core_identity=str(U(1)),
        core_intercept_worker_id=str(U(2)),
        core_migration_goal=(20, -8),
        core_migration_goal_kind="activity",
        core_last_move_delta=(1, 0),
        core_last_move_tick=123,
        dirty=True,
    )
    saved.save(force=True)

    restored = ScoutMemory(path=path)
    restored.load()
    assert restored.core_identity == str(U(1))
    assert restored.core_intercept_worker_id == str(U(2))
    assert restored.core_migration_goal == (20, -8)
    assert restored.core_migration_goal_kind == "activity"
    assert restored.core_last_move_delta == (1, 0)
    assert restored.core_last_move_tick == 123


def test_replacement_core_clears_persisted_migration_state():
    memory = ScoutMemory(
        core_identity=str(U(1)),
        core_intercept_worker_id=str(U(2)),
        core_migration_goal=(20, -8),
        core_migration_goal_kind="activity",
        core_last_move_delta=(1, 0),
        core_last_move_tick=123,
    )
    memory.sync_core_identity(str(U(99)))
    assert memory.core_identity == str(U(99))
    assert memory.core_intercept_worker_id is None
    assert memory.core_migration_goal is None
    assert memory.core_migration_goal_kind is None
    assert memory.core_last_move_delta is None
    assert memory.core_last_move_tick == -1


def test_scout_coverage_discards_the_oldest_cells_at_hard_limit(monkeypatch):
    import tactic

    monkeypatch.setattr(tactic, "SCOUT_COVERAGE_MAX_CELLS", 3)
    memory = ScoutMemory(
        scout_seen={(0, 0): 99, (1, 0): 100, (2, 0): 100, (3, 0): 100}
    )
    memory.expire(100)
    assert memory.scout_seen == {(1, 0): 100, (2, 0): 100, (3, 0): 100}
    assert memory.dirty is True


def test_legacy_relative_scout_state_is_cleared_but_obstacles_survive(tmp_path):
    path = tmp_path / "scout_state.json"
    path.write_text(
        '{"offsets":{"worker":7},"known_obstacles":[[4,5]],'
        '"scout_seen":{"1,0":99},"scout_targets":{"worker":[1,0]}}'
    )
    memory = ScoutMemory(path=path)
    memory.load()
    assert memory.known_obstacles == {(4, 5)}
    assert memory.offsets == {}
    assert memory.scout_seen == {}
    assert memory.scout_targets == {}
    assert memory.dirty is True


def test_schema_three_absolute_coverage_survives_metrics_upgrade(tmp_path):
    path = tmp_path / "scout_state.json"
    path.write_text(
        '{"schema_version":3,"known_obstacles":[[4,5]],'
        '"scout_seen":{"1,2":99},"scout_targets":{"worker":[1,2]}}'
    )
    memory = ScoutMemory(path=path)
    memory.load()
    assert memory.known_obstacles == {(4, 5)}
    assert memory.scout_seen == {(1, 2): 99}
    assert memory.scout_targets == {"worker": (1, 2)}


def test_resource_event_amounts_are_aggregated_for_tick_logs():
    memory = ScoutMemory()
    events = (
        ResolutionEvent(
            event_id=U(70),
            tick=99,
            event_type="HARVEST_SUCCEEDED",
            values={"amount": 2},
        ),
        ResolutionEvent(
            event_id=U(71),
            tick=99,
            event_type="DEPOSIT_SUCCEEDED",
            values={"amount": 1},
        ),
        ResolutionEvent(
            event_id=U(72),
            tick=99,
            event_type="WORKER_CARGO_DROPPED",
            values={"amount": 2},
        ),
    )
    turn = make_turn(objects=(core_view(), worker_view((3, 0), uid=2)), events=events)
    decide(turn, memory)
    assert memory.last_resource_flow == Counter(harvest=2, deposit=1, dropped=2)


def test_combat_events_are_aggregated_into_effectiveness_telemetry():
    from tactic import summarize_combat

    events = (
        ResolutionEvent(
            event_id=U(73),
            tick=99,
            event_type="SWEEP_RESOLVED",
            values={"targets_hit": 3},
        ),
        ResolutionEvent(
            event_id=U(74),
            tick=99,
            event_type="SHOT_HIT",
            values={"damage": 1},
        ),
        ResolutionEvent(
            event_id=U(75),
            tick=99,
            event_type="SHOT_MISSED",
            reason_code="SHOT_MISSED",
        ),
        ResolutionEvent(
            event_id=U(76),
            tick=99,
            event_type="DESTRUCTION_PARTICIPATION",
            reason_code="UNIT",
        ),
    )
    memory = ScoutMemory()
    turn = make_turn(objects=(core_view(), worker_view((3, 0), uid=2)), events=events)
    decide(turn, memory)
    assert memory.last_combat_results == Counter(
        sweeps=1,
        sweep_targets=3,
        ranger_hits=1,
        ranger_damage=1,
        ranger_misses=1,
        destructions=1,
    )
    assert summarize_combat(memory.last_combat_results) == (
        "combat[sweeps:1,targets:3,shots:1/1,damage:1,destructions:1]"
    )


def test_migration_telemetry_reports_ring_quota_and_goal_distance():
    from tactic import summarize_migration

    memory = ScoutMemory(
        core_migration_goal=(-768, 0),
        core_migration_goal_kind="density",
    )
    turn = make_turn(objects=(core_view(position=(-800, 0)), worker_view((-799, 0))))
    assert summarize_migration(turn, memory) == (
        "migration[ring:24,quota:4,kind:density,goal:(-768, 0),distance:32]"
    )


# --- P1: bounded terrain memory, durable cooldowns, honest deposit ETA -------


def test_obstacle_memory_is_bounded_and_keeps_the_cells_near_the_core():
    from tactic import OBSTACLE_MEMORY_KEEP_CELLS, OBSTACLE_MEMORY_MAX_CELLS

    # A column of cells walking away from the Core: index == distance.
    overflow = OBSTACLE_MEMORY_MAX_CELLS + 500
    memory = ScoutMemory(known_obstacles={(0, y) for y in range(1, overflow + 1)})
    dropped = memory.prune_obstacles((0, 0))
    assert dropped == overflow - OBSTACLE_MEMORY_KEEP_CELLS
    assert len(memory.known_obstacles) == OBSTACLE_MEMORY_KEEP_CELLS
    # The nearest cells survive and the most distant ones are gone.
    assert (0, 1) in memory.known_obstacles
    assert (0, OBSTACLE_MEMORY_KEEP_CELLS) in memory.known_obstacles
    assert (0, OBSTACLE_MEMORY_KEEP_CELLS + 1) not in memory.known_obstacles
    assert (0, overflow) not in memory.known_obstacles
    assert memory.dirty is True


def test_obstacle_memory_below_the_cap_is_never_pruned():
    memory = ScoutMemory(known_obstacles={(4, 5), (6, 7)}, dirty=False)
    assert memory.prune_obstacles((0, 0)) == 0
    assert memory.known_obstacles == {(4, 5), (6, 7)}
    assert memory.dirty is False


def test_learning_an_obstacle_enforces_the_cap():
    from tactic import OBSTACLE_MEMORY_KEEP_CELLS, OBSTACLE_MEMORY_MAX_CELLS

    memory = ScoutMemory(
        known_obstacles={(0, y) for y in range(1, OBSTACLE_MEMORY_MAX_CELLS + 1)}
    )
    turn = make_turn(
        objects=(
            core_view(),
            worker_view((1, 0), uid=2),
            terrain("OBSTACLE", [(2, 0)]),
        ),
    )
    decide(turn, memory)
    assert (2, 0) in memory.known_obstacles
    assert len(memory.known_obstacles) == OBSTACLE_MEMORY_KEEP_CELLS


def test_depleted_contested_and_resource_progress_survive_a_restart(tmp_path):
    from tactic import ResourceProgress

    path = tmp_path / "scout_state.json"
    saved = ScoutMemory(
        known_resources={(8, 9)},
        resource_assignments={str(U(2)): (8, 9)},
        resource_progress={str(U(2)): ResourceProgress((8, 9), 11, 3)},
        depleted={(2, 0): 140},
        contested={(1, 0): 104},
        path=path,
        dirty=True,
    )
    saved.save()
    restored = ScoutMemory(path=path)
    restored.load()
    assert restored.depleted == {(2, 0): 140}
    assert restored.contested == {(1, 0): 104}
    progress = restored.resource_progress[str(U(2))]
    assert progress.target == (8, 9)
    assert progress.best_cost == 11
    assert progress.stalled_ticks == 3


def test_restored_depleted_node_is_still_skipped_after_a_restart(tmp_path):
    from tactic import available_resources

    path = tmp_path / "scout_state.json"
    saved = ScoutMemory(
        known_resources={(2, 0), (9, 0)},
        depleted={(2, 0): 140},
        path=path,
        dirty=True,
    )
    saved.save()
    restored = ScoutMemory(path=path)
    restored.load()
    # A restart used to forget the cooldown and send a Worker straight back to
    # the node the server had just reported empty.
    turn = make_turn(objects=(core_view(), worker_view((1, 0), uid=2)), tick=100)
    assert available_resources(turn, restored) == {(9, 0)}


def test_expired_cooldowns_from_an_old_state_file_are_dropped(tmp_path):
    path = tmp_path / "scout_state.json"
    saved = ScoutMemory(
        depleted={(2, 0): 50},
        contested={(1, 0): 40},
        path=path,
        dirty=True,
    )
    saved.save()
    restored = ScoutMemory(path=path)
    restored.load()
    restored.expire(100)
    assert restored.depleted == {}
    assert restored.contested == {}


def test_schema_four_state_loads_without_the_new_cooldown_fields(tmp_path):
    path = tmp_path / "scout_state.json"
    path.write_text(
        '{"schema_version":4,"known_obstacles":[[4,5]],"scout_seen":{"1,2":99}}'
    )
    memory = ScoutMemory(path=path)
    memory.load()
    assert memory.known_obstacles == {(4, 5)}
    assert memory.scout_seen == {(1, 2): 99}
    assert memory.depleted == {}
    assert memory.contested == {}
    assert memory.resource_progress == {}


def test_deposit_eta_is_unchanged_when_no_wall_intervenes():
    from tactic import nearest_deposit_eta

    wall = frozenset({(5, 5), (6, 6)})
    turn = make_turn(
        resources=0,
        objects=(core_view(position=(0, 0)), worker_view((1, 0), cargo=1, uid=2)),
    )
    assert nearest_deposit_eta(turn) == 1
    assert nearest_deposit_eta(turn, wall) == 1


def test_deposit_eta_reports_the_detour_a_wall_forces():
    from tactic import nearest_deposit_eta

    # A vertical wall between the Worker and the Core, open only beyond y=2.
    wall = frozenset({(1, y) for y in range(-2, 3)})
    turn = make_turn(
        resources=0,
        objects=(core_view(position=(0, 0)), worker_view((2, 0), cargo=1, uid=2)),
    )
    assert nearest_deposit_eta(turn) == 2
    detour = nearest_deposit_eta(turn, wall)
    assert detour is not None and detour > 2


def test_deposit_eta_prefers_the_worker_with_the_shortest_route():
    from tactic import nearest_deposit_eta

    # The Manhattan-nearest Worker is walled off; a farther one has a clear run.
    wall = frozenset({(1, y) for y in range(-4, 5)})
    turn = make_turn(
        resources=0,
        objects=(
            core_view(position=(0, 0)),
            worker_view((2, 0), cargo=1, uid=2),
            worker_view((0, -5), cargo=1, uid=3),
        ),
    )
    assert nearest_deposit_eta(turn, wall) == 5


def test_corrupt_cooldown_and_progress_entries_are_skipped(tmp_path):
    import json

    path = tmp_path / "scout_state.json"
    raw = {
        "schema_version": 5,
        "depleted": {"2,0": 140, "nocomma": 9, "4,0": True},
        "contested": {"1,0": 104},
        "resource_progress": {
            "keep": [8, 9, 11, 3],
            "short": [8, 9, 11],
            "notalist": "x",
            "nonint": [8, 9, 11, "z"],
        },
    }
    path.write_text(json.dumps(raw))
    memory = ScoutMemory(path=path)
    memory.load()
    assert memory.depleted == {(2, 0): 140}
    assert memory.contested == {(1, 0): 104}
    assert set(memory.resource_progress) == {"keep"}


# --- loitering: a Worker frozen on one cell must become visible and unstick ---


def test_standing_still_is_counted_as_a_stall():
    from tactic import WORKER_STALL_TICKS

    memory = ScoutMemory()
    worker_id = str(U(2))
    # A real route first, so the oscillation window looks perfectly healthy.
    for cell in ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0)):
        memory.record_position(worker_id, cell)
    assert memory.is_looping(worker_id) is False
    for _ in range(WORKER_STALL_TICKS - 1):
        memory.record_position(worker_id, (6, 0))
    assert memory.is_looping(worker_id) is False
    memory.record_position(worker_id, (6, 0))
    assert memory.is_looping(worker_id) is True


def test_moving_again_clears_the_stall():
    from tactic import WORKER_STALL_TICKS

    memory = ScoutMemory()
    worker_id = str(U(2))
    memory.record_position(worker_id, (6, 0))
    for _ in range(WORKER_STALL_TICKS + 4):
        memory.record_position(worker_id, (6, 0))
    assert memory.is_looping(worker_id) is True
    memory.record_position(worker_id, (7, 0))
    assert memory.position_stalls.get(worker_id, 0) == 0
    assert memory.is_looping(worker_id) is False


def test_dead_workers_do_not_keep_a_stall_counter():
    memory = ScoutMemory()
    memory.record_position(str(U(2)), (6, 0))
    memory.record_position(str(U(2)), (6, 0))
    memory.record_position(str(U(3)), (1, 1))
    memory.record_position(str(U(3)), (1, 1))
    memory.prune_workers({str(U(3))})
    assert set(memory.position_stalls) == {str(U(3))}


def test_stall_counters_survive_a_restart(tmp_path):
    path = tmp_path / "scout_state.json"
    saved = ScoutMemory(
        scout_positions={str(U(2)): [(6, 0)]},
        position_stalls={str(U(2)): 5},
        path=path,
        dirty=True,
    )
    saved.save()
    restored = ScoutMemory(path=path)
    restored.load()
    # A restart used to reset the counter, so a Worker that had already idled for
    # minutes got a fresh clean slate and kept idling.
    assert restored.position_stalls == {str(U(2)): 5}


def _boxed_scout(stalls: int):
    """A scout whose every route is soft-blocked, with open terrain underneath."""

    from tactic import MovementReservations, TickBudget, plan_worker

    turn = make_turn(
        resources=0,
        objects=(core_view(position=(0, 0)), worker_view((10, 10), uid=2)),
    )
    worker = turn.workers[0]
    # Set the history directly: record_position() clears the counter whenever it
    # appends a new cell, which would undo the standstill this test is about.
    memory = ScoutMemory(
        scout_positions={str(worker.id): [(10, 10)]},
        position_stalls={str(worker.id): stalls},
    )
    soft = {(9, 10), (11, 10), (10, 9), (10, 11)}
    plan_worker(
        worker,
        turn,
        (0, 0),
        (0, 0),
        True,
        soft,
        MovementReservations(),
        None,
        frozenset(),
        False,
        TickBudget(resources=0, space=10),
        memory,
        None,
        hard_blocked=set(),
    )
    return turn


def test_idle_scout_unsticks_by_dropping_threat_arcs():
    from tactic import WORKER_STALL_TICKS

    turn = _boxed_scout(WORKER_STALL_TICKS)
    # Before the fix this Worker queued Wait every Tick forever: threat arcs
    # sealed every route and nothing ever noticed it had stopped moving.
    assert action_type(turn.plan, 2) == "MoveAction"
    assert direction_of(turn.plan, 2) in (Direction.LEFT, Direction.UP)


def test_scout_that_has_not_idled_long_enough_still_respects_threat_arcs():
    from tactic import WORKER_STALL_TICKS

    turn = _boxed_scout(WORKER_STALL_TICKS - 2)
    assert action_type(turn.plan, 2) == "WaitAction"


# --- unit preservation: hurt or cornered Workers walk home ------------------


def test_healthy_worker_walks_to_its_node():
    turn = make_turn(
        resources=5,
        objects=(
            core_view(position=(0, 0)),
            worker_view((10, 0), cargo=0, hp=2, uid=2),
            terrain("RESOURCE", [(12, 0)]),
        ),
    )
    decide(turn, ScoutMemory())
    assert direction_of(turn.plan, 2) is Direction.RIGHT


def test_damaged_worker_abandons_the_node_and_walks_home():
    turn = make_turn(
        resources=5,
        objects=(
            core_view(position=(0, 0)),
            worker_view((10, 0), cargo=0, hp=1, uid=2),
            terrain("RESOURCE", [(12, 0)]),
        ),
    )
    decide(turn, ScoutMemory())
    # Damage used to change nothing at all: this Worker kept walking outward on
    # its last hit point and could never reach the only cell that heals it.
    assert direction_of(turn.plan, 2) is Direction.LEFT


def test_worker_retreats_from_a_nearby_enemy():
    from tactic import WORKER_FLEE_DISTANCE

    turn = make_turn(
        resources=5,
        objects=(
            core_view(position=(0, 0)),
            worker_view((10, 0), cargo=0, hp=2, uid=2),
            terrain("RESOURCE", [(12, 0)]),
            enemy_view((10, WORKER_FLEE_DISTANCE), uid=90),
        ),
    )
    decide(turn, ScoutMemory())
    # The Core is twelve cells away, so core-level danger never fired and this
    # Worker used to walk on toward the node with an enemy on top of it.
    assert direction_of(turn.plan, 2) is Direction.LEFT


def test_worker_ignores_an_enemy_that_is_far_away():
    from tactic import WORKER_FLEE_DISTANCE

    turn = make_turn(
        resources=5,
        objects=(
            core_view(position=(0, 0)),
            worker_view((10, 0), cargo=0, hp=2, uid=2),
            terrain("RESOURCE", [(12, 0)]),
            enemy_view((10, WORKER_FLEE_DISTANCE + 6), uid=90),
        ),
    )
    decide(turn, ScoutMemory())
    assert direction_of(turn.plan, 2) is Direction.RIGHT


def test_damaged_worker_on_the_core_cell_is_finally_healed():
    turn = make_turn(
        resources=5,
        objects=(
            core_view(position=(0, 0)),
            worker_view((0, 0), cargo=0, hp=1, uid=2),
        ),
    )
    memory = ScoutMemory()
    decide(turn, memory)
    # The heal path had never once fired in production: nothing sent a damaged
    # Worker to the one cell where healing is allowed.
    assert action_type(turn.plan, 2) == "HealAction"
    assert memory.last_intents["healing"] == 1


def test_healing_is_skipped_when_the_core_cannot_afford_it():
    turn = make_turn(
        resources=2,
        objects=(
            core_view(position=(0, 0)),
            worker_view((0, 0), cargo=0, hp=1, uid=2),
        ),
    )
    decide(turn, ScoutMemory())
    assert action_type(turn.plan, 2) != "HealAction"


def test_retreating_worker_still_prefers_a_safe_route():
    # A Vanguard threatens only its four neighbours, so parking it at (8, 0)
    # poisons exactly the next step of the direct route home. Fleeing must mean
    # detouring around the firing line, not walking through it.
    turn = make_turn(
        resources=5,
        objects=(
            core_view(position=(0, 0)),
            worker_view((10, 0), cargo=0, hp=1, uid=2),
            enemy_view((8, 0), uid=90),
        ),
    )
    decide(turn, ScoutMemory())
    assert action_type(turn.plan, 2) == "MoveAction"
    assert direction_of(turn.plan, 2) is not Direction.LEFT


def test_retreating_worker_does_not_block_the_emergency_defender():
    # The Core needs the spawn cell far more than a two hit point Worker needs
    # cover: spawn_cell_open() is false while any Unit is projected onto it.
    turn = make_turn(
        resources=12,
        objects=(
            core_view(position=(0, 0)),
            worker_view((1, 0), cargo=0, hp=2, uid=2),
            enemy_view((2, 0), uid=90),
        ),
    )
    decide(turn, ScoutMemory())
    assert core_action_type(turn) == "SpawnAction"
    assert turn.plan.core_action.unit_type is not UnitType.WORKER


def test_non_guard_worker_does_not_cross_core_and_block_emergency_spawn():
    from tactic import projected_unit_position

    turn = make_turn(
        resources=12,
        objects=(
            core_view(position=(0, 0), shield=2),
            worker_view((-1, 0), uid=2),
            worker_view((0, -1), uid=3),
            terrain("RESOURCE", [(0, 2)]),
            enemy_view((2, 0), uid=90),
        ),
    )
    decide(turn, ScoutMemory())
    assert core_action_type(turn) == "SpawnAction"
    assert projected_unit_position(turn.unit(U(3)), turn.plan) != (0, 0)


def test_cargo_worker_evacuates_core_cell_for_emergency_spawn():
    turn = make_turn(
        resources=12,
        objects=(
            core_view(position=(0, 0), shield=2),
            worker_view((0, 0), cargo=1, uid=2),
            enemy_view((2, 0), uid=90),
        ),
    )
    memory = ScoutMemory()
    decide(turn, memory)
    assert action_type(turn.plan, 2) == "MoveAction"
    assert memory.last_intents["evacuating"] == 1
    assert core_action_type(turn) == "SpawnAction"


def test_contestable_core_departure_does_not_enable_emergency_spawn():
    turn = make_turn(
        resources=12,
        objects=(
            core_view(position=(0, 0), shield=2),
            worker_view((0, 0), cargo=1, uid=2),
            enemy_view((2, 0), uid=90),
            enemy_view((1, -1), uid=91, unit_type=UnitType.WORKER),
        ),
    )
    memory = ScoutMemory()
    decide(turn, memory)
    assert action_type(turn.plan, 2) == "MoveAction"
    assert direction_of(turn.plan, 2) is Direction.UP
    assert core_action_type(turn) != "SpawnAction"
    assert memory.last_defense_status is not None
    assert "spawn_blockers:spawn_cell" in memory.last_defense_status


def test_underfunded_core_accepts_cargo_then_clears_for_spawn_next_tick():
    memory = ScoutMemory()
    deposit_turn = make_turn(
        tick=100,
        resources=8,
        objects=(
            core_view(position=(0, 0), shield=2),
            worker_view((0, 0), cargo=2, uid=2),
            enemy_view((2, 0), uid=90),
        ),
    )
    decide(deposit_turn, memory)
    assert action_type(deposit_turn.plan, 2) == "DepositAction"
    assert core_action_type(deposit_turn) == "WaitAction"

    funded_turn = make_turn(
        tick=101,
        resources=10,
        objects=(
            core_view(position=(0, 0), shield=2),
            worker_view((0, 0), cargo=0, uid=2),
            enemy_view((1, 0), uid=90),
        ),
    )
    decide(funded_turn, memory)
    assert action_type(funded_turn.plan, 2) == "MoveAction"
    assert core_action_type(funded_turn) == "SpawnAction"


def test_underfunded_core_allows_adjacent_cargo_worker_to_enter():
    turn = make_turn(
        resources=8,
        objects=(
            core_view(position=(0, 0), shield=2),
            worker_view((-1, 0), cargo=2, uid=2),
            enemy_view((2, 0), uid=90),
        ),
    )
    decide(turn, ScoutMemory())
    assert action_type(turn.plan, 2) == "MoveAction"
    assert direction_of(turn.plan, 2) is Direction.RIGHT
    assert core_action_type(turn) == "WaitAction"


def test_core_danger_does_not_guard_from_the_wrong_approach_side():
    memory = ScoutMemory()
    turn = make_turn(
        resources=8,
        objects=(
            core_view(position=(0, 0), shield=2),
            worker_view((-2, 0), uid=2),
            worker_view((0, 5), uid=3),
            worker_view((1, 5), uid=4),
            worker_view((-1, 5), uid=5),
            worker_view((0, 6), uid=6),
            worker_view((1, 6), uid=7),
            enemy_view((2, 0), uid=90),
        ),
    )
    decide(turn, memory)
    assert memory.last_intents["guarding"] == 0
    assert memory.last_intents["scouting"] == 5
    assert memory.last_intents["evading"] == 1
    assert direction_of(turn.plan, 2) is Direction.RIGHT


def test_core_danger_holds_one_worker_already_blocking_vanguard_approach():
    memory = ScoutMemory()
    turn = make_turn(
        resources=8,
        objects=(
            core_view(position=(0, 0), shield=2),
            worker_view((1, 0), uid=2),
            worker_view((0, 5), uid=3),
            enemy_view((3, 0), uid=90),
        ),
    )
    decide(turn, memory)
    assert memory.last_intents["guarding"] == 1
    assert action_type(turn.plan, 2) == "WaitAction"
