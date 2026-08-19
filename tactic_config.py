"""Policy constants shared by the Arena Hero tactic modules.

Keeping tuning values in this dependency-leaf module lets state persistence and
the live planner evolve independently without creating circular imports.
"""

from arena_hero import Direction, UnitType


DIRECTION_ORDER = (Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT)
# ``Direction.delta`` is an SDK property. Cache it once instead of rebuilding
# the same tiny mapping inside every A* expansion and planning loop.
DIRECTION_DELTAS = {direction: direction.delta for direction in DIRECTION_ORDER}
MIN_ECONOMY_WORKERS = 4
MAX_WORKER_POPULATION = 14
MIN_RANGER_POPULATION = 2
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
# useful time. Leg limits protect pathfinding; the adaptive round-trip budget
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
# scout spread over four times the area covers it four times more slowly. Grow
# the search only after most of the near grid has actually been seen, and keep
# the recall tether the same distance beyond whichever radius is in force.
SCOUT_EXHAUSTED_RATIO = 0.8
SCOUT_FAR_DISTANCE = 72
SCOUT_TETHER_MARGIN = SCOUT_TETHER_DISTANCE - SCOUT_MAX_DISTANCE
SCOUT_SAFE_RETURN_DISTANCE = 12
SCOUT_SECTOR_COUNT = 4
# Keep most Workers close enough to refresh nearby resource visibility while a
# smaller stable group searches beyond the local disc.  At the 14-Worker cap
# this assigns 8 local and 6 remote roles.
SCOUT_REMOTE_ROLE_NUMERATOR = 3
SCOUT_REMOTE_ROLE_DENOMINATOR = 7
SCOUT_COVERAGE_TTL = 4096
SCOUT_COVERAGE_MAX_CELLS = 32768
SCOUT_HISTORY_LIMIT = 8
SCOUT_LOOP_WINDOW = 6
# A Worker standing perfectly still appends nothing to its position history, so
# oscillation detection alone cannot see it. Count the idle Ticks separately.
WORKER_STALL_TICKS = 6
SCOUT_ABSOLUTE_GRID_SCHEMA = 3
SCOUT_STATE_SCHEMA = 8
SCOUT_SAVE_INTERVAL = 8
# Obstacle terrain is permanent, so it is never expired by age. It is still
# bounded, because a long game walks the Core far enough to accumulate terrain
# that no longer influences any route. Pruning drops the cells farthest from
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
# Ticks after the shooter drops out of sight. Without that hysteresis a loaded
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
CORE_MIGRATION_REVERSE_COOLDOWN_TICKS = 12
CORE_RELOCATION_MIN_WORKERS = 4
CORE_RELOCATION_MIN_ACTIVE_WORKERS = 2
CORE_RELOCATION_ENEMY_BUFFER = 8
CORE_MIGRATION_PRODUCTION_GAP = 2
CORE_DEFENSE_ALERT_DISTANCE = 8
CORE_SHIELD_EMERGENCY_FLOOR = 2
CORE_THREAT_CAUTION_TICKS = 6
CORE_PREFERRED_RESOURCE_QUOTA = 6
CORE_DENSITY_MILESTONE_DISTANCE = 32
# A Worker has two hit points and sees three cells, so it can neither win nor
# outrun a fight it has already walked into. Leave before contact, not after.
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

# General pursuit is intentionally smaller than the scouting disc.  Immediate
# Core threats still receive the whole defense force, while ordinary visible
# targets can draw only a pair of hunters while they remain inside the Core's
# pursuit radius, so a retreating target cannot lure those hunters outward.
COMBAT_PURSUIT_DISTANCE = 16
COMBAT_PURSUIT_HUNTERS = 2

# Event types carrying one of these markers mean the server rejected or failed
# something we asked for, which is always worth surfacing in the log.
FAILURE_MARKERS = ("FAILED", "REJECTED", "INVALID", "DENIED", "MISSED", "OVERFLOW")
STALE_RESOURCE_REASONS = {"RESOURCE_DEPLETED", "NOT_RESOURCE_CELL", "NODE_EXHAUSTED"}

__all__ = [name for name in globals() if name.isupper()]
