"""Read-only snapshot of one Turn: what the bot sees and what it would do.

Connects, takes the first Turn, prints the observable world plus the plan the
current tactic would queue, and exits. It never calls submit(), so it is safe
to run while the supervisor is playing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from arena_hero import ArenaHeroClient

from tactic import ScoutMemory, decide, load_api_key


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="/root/arena-hero/.env")
    parser.add_argument("--state", default="/root/arena-hero/scout_state.json")
    args = parser.parse_args()

    memory = ScoutMemory(path=Path(args.state))
    memory.load()

    with ArenaHeroClient(api_key=load_api_key(args.env)) as game:
        turn = next(game.turns())

        print(f"tick {turn.tick}  status {turn.state.status.value}")
        print(
            f"resources {turn.resources}/{turn.resource_capacity} "
            f"(space {turn.resource_space})  population {turn.state.population}"
        )

        core = turn.core
        print(
            f"core {tuple(core.position)} hp {core.hp} shield {core.shield} "
            f"state {core.view.state.value}"
            if core is not None
            else "core: not controlled"
        )

        beacon = turn.beacon
        print(
            f"beacon {tuple(beacon.position)} status {beacon.status} "
            f"carrier {str(beacon.carrier_id)[:8] if beacon.carrier_id else None}"
        )

        print(f"units ({len(turn.units)}):")
        for unit in sorted(turn.units, key=lambda u: str(u.id)):
            print(
                f"  {str(unit.id)[:8]} {unit.unit_type.value:<8} {tuple(unit.position)} "
                f"hp {unit.hp} cargo {getattr(unit, 'cargo', None)}"
            )

        resources = sorted(turn.resource_cells)
        origin = core.position if core is not None else (0, 0)
        print(f"resource cells ({len(resources)}): {resources[:12]}")
        if resources:
            nearest = min(
                resources,
                key=lambda c: abs(c[0] - origin[0]) + abs(c[1] - origin[1]),
            )
            distance = abs(nearest[0] - origin[0]) + abs(nearest[1] - origin[1])
            print(f"  nearest to core: {nearest} at distance {distance}")
        else:
            print("  none visible -> workers are scouting")
        print(f"obstacle cells: {len(turn.obstacle_cells)}")
        print(
            "visible enemies:",
            [(e.kind, tuple(e.position)) for e in turn.visible_enemies] or "none",
        )

        print(f"events from last tick ({len(turn.events)}):")
        for event in turn.events:
            print(f"  {event.event_type} reason={event.reason_code} values={event.values}")

        decide(turn, memory)
        plan = turn.plan
        print("would queue (NOT submitted):")
        for unit_id, action in sorted(plan.unit_actions.items(), key=lambda kv: str(kv[0])):
            print(
                f"  {str(unit_id)[:8]} {type(action).__name__}",
                getattr(action, "direction", ""),
            )
        print(
            "  core:",
            type(plan.core_action).__name__ if plan.core_action else None,
            getattr(plan.core_action, "unit_type", ""),
        )


if __name__ == "__main__":
    main()
