import math

import numpy as np
import pytest

from app.models.enums import RouteProfile, WaypointType
from app.processing.hazard import CATEGORY_UNKNOWN
from app.routing.astar import SearchLimits, SearchSpace, SearchStatus, astar, octile, validate_cell_path
from app.routing.cost_model import (
    DEFAULT_ROUTING_CONFIG, RoutingConfig, SafetyWeights, SpeedModel, cell_factors, speed_mps, uncertainty_field,
)
from app.routing.grid import RoutingConstraints, build_routing_grid
from app.routing.planner import (
    Stop, activity_seconds, evaluate_geometry, plan_profile, stop_arrival_times,
)
from tests.routing_fixtures import dijkstra_oracle, make_routing_grid, stop_at

SHORT, SAFE = RouteProfile.SHORTER_DISTANCE, RouteProfile.SAFETY_FIRST
CONFIG = DEFAULT_ROUTING_CONFIG
SQRT2 = math.sqrt(2)


def search(grid, start, goal, profile=SHORT, **kwargs):
    space = SearchSpace(grid, cell_factors(grid, profile, CONFIG))
    return astar(space, start, goal, **kwargs)


def plan(grid, stops, profile=SHORT, **kwargs):
    return plan_profile(grid, stops, profile, CONFIG, **kwargs)


# --- feasible routes: exact, hand-verifiable distances -------------------------------------------

@pytest.mark.parametrize("goal", [(2, 10), (7, 0), (7, 5), (7, 10), (0, 3)])
def test_open_field_distance_is_exact_and_geometry_matches(goal):
    grid = make_routing_grid(["." * 12] * 8)
    result = plan(grid, [stop_at(grid, 2, 0), stop_at(grid, *goal)])
    dr, dc = abs(goal[0] - 2), goal[1]
    expected = 10.0 * (max(dr, dc) + (SQRT2 - 1) * min(dr, dc))       # octile distance, hand-computed
    assert result.status == SearchStatus.FOUND
    assert result.metrics.distance_m == pytest.approx(expected)
    assert result.metrics.total_cost == pytest.approx(expected)          # shortest profile: cost == length


def test_reported_distance_equals_independent_polyline_length():
    grid = make_routing_grid(["." * 15] * 10)
    result = plan(grid, [stop_at(grid, 1, 1), stop_at(grid, 8, 13)])
    xy = np.array(result.points)
    independent = float(np.sum(np.hypot(*np.diff(xy, axis=0).T)))
    assert result.metrics.distance_m == pytest.approx(independent, rel=1e-12)
    assert evaluate_geometry(grid, result.points, SHORT, CONFIG).distance_m == pytest.approx(independent)


def test_start_and_end_are_the_exact_requested_coordinates_not_cell_centres():
    grid = make_routing_grid(["." * 10] * 5)
    cx, cy = grid.grid.cell_center(2, 1)
    start = Stop(WaypointType.START, "start", cx + 3.0, cy - 2.0)
    end = Stop(WaypointType.HABITAT_RETURN, "end", cx + 50.0 + 4.0, cy + 1.0)
    result = plan(grid, [start, end])
    assert result.points[0] == (start.x_m, start.y_m) and result.points[-1] == (end.x_m, end.y_m)
    xy = np.array(result.points)
    assert result.metrics.distance_m == pytest.approx(float(np.hypot(*np.diff(xy, axis=0).T).sum()))
    centre_to_centre = 50.0
    assert result.metrics.distance_m != pytest.approx(centre_to_centre)   # offsets are included


def test_start_equals_goal_is_a_zero_length_route():
    grid = make_routing_grid(["....."] * 3)
    result = plan(grid, [stop_at(grid, 1, 1), stop_at(grid, 1, 1)])
    assert result.status == SearchStatus.FOUND and result.metrics.distance_m == 0.0
    assert result.metrics.travel_time_s == 0.0 and len(result.points) == 1


# --- obstacles, corners and detours ---------------------------------------------------------------

WALL_WITH_GAP = [
    "..#..",
    "..#..",
    "..#..",
    "..#..",
    ".....",
]


def test_path_detours_through_the_gap_with_exact_length():
    grid = make_routing_grid(WALL_WITH_GAP)
    result = plan(grid, [stop_at(grid, 0, 0), stop_at(grid, 0, 4)])
    assert result.status == SearchStatus.FOUND
    # Corner cutting is forbidden, so each side needs 4 + sqrt(2) cells (see test below for the unrestricted value).
    assert result.metrics.distance_m == pytest.approx(10.0 * (8 + 2 * SQRT2))
    cells = [grid.cell_of(*p) for p in result.points]
    assert (4, 2) in cells and all(not grid.blocked[c] for c in cells)


def test_diagonal_moves_never_cut_blocked_corners():
    grid = make_routing_grid(WALL_WITH_GAP)
    cells = search(grid, (0, 0), (0, 4)).cells
    assert ((3, 1), (4, 2)) not in list(zip(cells, cells[1:]))
    assert ((3, 3), (4, 2)) not in list(zip(cells, cells[1:]))
    validate_cell_path(grid, cells)
    # Without the rule the route would be shorter: 4*sqrt(2) + 4 cells. The rule costs extra length.
    assert search(grid, (0, 0), (0, 4)).cost > 10.0 * (4 * SQRT2 + 4)


def test_diagonal_gap_between_two_blocked_cells_is_not_passable():
    grid = make_routing_grid([
        "..#",
        "..#",
        "##.",
    ])
    # (1,1) and (2,2) touch only diagonally with blocked (1,2) and (2,1) at the corner
    result = search(grid, (0, 0), (2, 2))
    assert result.status == SearchStatus.UNREACHABLE


def test_validate_cell_path_rejects_bad_paths():
    grid = make_routing_grid(WALL_WITH_GAP)
    with pytest.raises(ValueError, match="blocked"):
        validate_cell_path(grid, [(0, 0), (0, 1), (0, 2)])
    with pytest.raises(ValueError, match="contiguous"):
        validate_cell_path(grid, [(0, 0), (0, 1), (2, 1)])
    with pytest.raises(ValueError, match="corner"):
        validate_cell_path(grid, [(3, 1), (4, 2)])
    with pytest.raises(ValueError, match="empty"):
        validate_cell_path(grid, [])


# --- blocked and unreachable ---------------------------------------------------------------------

def test_fully_walled_destination_is_unreachable_with_no_geometry_or_metrics():
    grid = make_routing_grid([
        ".....",
        ".###.",
        ".#.#.",
        ".###.",
        ".....",
    ])
    result = plan(grid, [stop_at(grid, 0, 0, "start"), stop_at(grid, 2, 2, "inside")])
    assert result.status == SearchStatus.UNREACHABLE
    assert result.failed_leg == 0 and result.points == [] and result.metrics is None
    assert result.legs[0].expanded > 0


def test_unreachable_when_start_is_enclosed_too():
    grid = make_routing_grid(["..#..", "..#..", "..#.."])
    assert search(grid, (0, 0), (0, 4)).status == SearchStatus.UNREACHABLE


def test_blocked_start_and_blocked_goal_are_reported_distinctly():
    grid = make_routing_grid(["..#..", ".....", "..?.."])
    assert search(grid, (0, 2), (1, 4)).status == SearchStatus.START_BLOCKED
    assert search(grid, (1, 0), (0, 2)).status == SearchStatus.GOAL_BLOCKED
    assert search(grid, (1, 0), (2, 2)).status == SearchStatus.GOAL_BLOCKED     # unknown terrain is blocked


def test_failed_middle_leg_reports_which_leg_failed():
    grid = make_routing_grid(["..#..", "..#..", "..#.."])
    stops = [stop_at(grid, 0, 0, "a"), stop_at(grid, 2, 1, "b"), stop_at(grid, 0, 4, "c")]
    result = plan(grid, stops)
    assert result.status == SearchStatus.UNREACHABLE and result.failed_leg == 1
    assert len(result.legs) == 2 and result.legs[0].status == SearchStatus.FOUND


# --- optimality against an independent oracle -----------------------------------------------------

@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("profile", [SHORT, SAFE])
def test_astar_cost_equals_independent_dijkstra(seed, profile):
    rng = np.random.default_rng(seed)
    rows, cols = 14, 14
    ascii_map = ["".join(rng.choice(list(".....#~"), cols)) for _ in range(rows)]
    ascii_map[0] = "." + ascii_map[0][1:]
    ascii_map[-1] = ascii_map[-1][:-1] + "."
    grid = make_routing_grid(ascii_map, slope=rng.uniform(0, 14, (rows, cols)).astype(np.float32))
    factors = cell_factors(grid, profile, CONFIG)
    expected = dijkstra_oracle(grid, factors, (0, 0), (rows - 1, cols - 1))
    with_h = search(grid, (0, 0), (rows - 1, cols - 1), profile)
    without_h = search(grid, (0, 0), (rows - 1, cols - 1), profile, use_heuristic=False)
    if math.isinf(expected):
        assert with_h.status == without_h.status == SearchStatus.UNREACHABLE
    else:
        assert with_h.status == SearchStatus.FOUND
        assert with_h.cost == pytest.approx(expected, rel=1e-9) and without_h.cost == pytest.approx(expected, rel=1e-9)
        validate_cell_path(grid, with_h.cells)


def test_heuristic_prunes_the_search_on_open_ground():
    grid = make_routing_grid(["." * 40] * 40)
    with_h = search(grid, (20, 0), (20, 39))
    without_h = search(grid, (20, 0), (20, 39), use_heuristic=False)
    assert with_h.cost == pytest.approx(without_h.cost) == pytest.approx(390.0)
    assert with_h.expanded * 5 < without_h.expanded


def test_heuristic_is_admissible_and_consistent():
    rng = np.random.default_rng(7)
    ascii_map = ["".join(rng.choice(list("...~#"), 12)) for _ in range(12)]
    grid = make_routing_grid(ascii_map, slope=rng.uniform(0, 14, (12, 12)).astype(np.float32))
    for profile in (SHORT, SAFE):
        assert cell_factors(grid, profile, CONFIG).min() >= 1.0        # every step costs at least its length
    for _ in range(300):
        a = (int(rng.integers(12)), int(rng.integers(12)))
        goal = (int(rng.integers(12)), int(rng.integers(12)))
        dr, dc = int(rng.integers(-1, 2)), int(rng.integers(-1, 2))
        b = (a[0] + dr, a[1] + dc)
        if (dr, dc) == (0, 0) or not (0 <= b[0] < 12 and 0 <= b[1] < 12):
            continue
        step = 10.0 * (SQRT2 if dr and dc else 1.0)
        assert octile(a[0] - goal[0], a[1] - goal[1], 10.0) <= step + octile(b[0] - goal[0], b[1] - goal[1], 10.0) + 1e-9


# --- shorter distance vs safety-first --------------------------------------------------------------

RISK_BAND = [
    "...............",
    "...............",
    "......~~~......",
    "......~~~......",
    "......~~~......",
    "......~~~......",
    "......~~~......",
    "...............",
    "...............",
]


def test_profiles_choose_different_routes_and_metrics_come_from_the_paths():
    grid = make_routing_grid(RISK_BAND)
    stops = [stop_at(grid, 4, 0), stop_at(grid, 4, 14)]
    short, safe = plan(grid, stops, SHORT), plan(grid, stops, SAFE)
    assert short.status == safe.status == SearchStatus.FOUND
    assert short.metrics.distance_m == pytest.approx(140.0)                # straight through the band
    assert safe.metrics.distance_m > short.metrics.distance_m              # the detour is longer
    assert safe.metrics.risk_score < short.metrics.risk_score              # ...and safer, per its own geometry
    assert safe.metrics.peak_risk < short.metrics.peak_risk and safe.metrics.high_risk_distance_m == 0.0
    assert short.metrics.high_risk_distance_m > 0
    assert short.points != safe.points


def test_each_profile_is_optimal_for_its_own_cost_function():
    grid = make_routing_grid(RISK_BAND)
    stops = [stop_at(grid, 4, 0), stop_at(grid, 4, 14)]
    short, safe = plan(grid, stops, SHORT), plan(grid, stops, SAFE)
    # Evaluate both geometries under both cost functions: neither can beat the profile optimised for it.
    assert evaluate_geometry(grid, short.points, SHORT, CONFIG).total_cost <= evaluate_geometry(grid, safe.points, SHORT, CONFIG).total_cost
    assert evaluate_geometry(grid, safe.points, SAFE, CONFIG).total_cost <= evaluate_geometry(grid, short.points, SAFE, CONFIG).total_cost
    assert short.metrics.distance_m <= safe.metrics.distance_m


def test_safety_first_equals_shortest_when_there_is_no_risk():
    grid = make_routing_grid(["." * 12] * 8)
    stops = [stop_at(grid, 0, 0), stop_at(grid, 7, 11)]
    short, safe = plan(grid, stops, SHORT), plan(grid, stops, SAFE)
    assert safe.metrics.distance_m == pytest.approx(short.metrics.distance_m)
    assert safe.metrics.total_cost == pytest.approx(safe.metrics.distance_m)


def test_safety_first_accepts_risk_when_no_alternative_exists():
    narrow = ["#######", "...~...", "#######"]
    grid = make_routing_grid(narrow)
    result = plan(grid, [stop_at(grid, 1, 0), stop_at(grid, 1, 6)], SAFE)
    assert result.status == SearchStatus.FOUND and result.metrics.peak_risk == pytest.approx(0.8)


def test_slope_penalty_steers_the_safety_route_toward_gentle_ground():
    slope = np.zeros((9, 15), np.float32)
    slope[3:6, 5:10] = 14.0                                                # steep but traversable plateau
    grid = make_routing_grid(["." * 15] * 9, slope=slope)
    stops = [stop_at(grid, 4, 0), stop_at(grid, 4, 14)]
    short, safe = plan(grid, stops, SHORT), plan(grid, stops, SAFE)
    assert short.metrics.max_slope_deg == 14.0 and safe.metrics.max_slope_deg < 14.0
    assert safe.metrics.distance_m > short.metrics.distance_m
    assert safe.metrics.mean_slope_deg < short.metrics.mean_slope_deg


# --- uncertainty -----------------------------------------------------------------------------------

def test_uncertainty_field_decays_from_unknown_terrain():
    unknown = np.zeros((9, 9), bool)
    unknown[4, 4] = True
    field = uncertainty_field(unknown, 10.0, 30.0)                         # 3 cells
    assert field[4, 4] == 0.0                                              # the unknown cell itself is blocked anyway
    assert field[4, 5] == 1.0 and field[3, 3] == 1.0
    assert field[4, 6] == pytest.approx(2 / 3) and field[4, 7] == pytest.approx(1 / 3) and field[4, 8] == 0.0
    assert uncertainty_field(unknown, 10.0, 0.0).max() == 0.0


def test_safety_route_keeps_away_from_unknown_terrain_when_it_can():
    ascii_map = ["." * 15] * 9
    ascii_map[4] = "." * 7 + "?" + "." * 7
    grid = make_routing_grid(ascii_map, buffer_m=30.0)
    stops = [stop_at(grid, 4, 0), stop_at(grid, 4, 14)]
    short, safe = plan(grid, stops, SHORT), plan(grid, stops, SAFE)
    assert short.metrics.uncertain_distance_m > safe.metrics.uncertain_distance_m
    assert safe.metrics.distance_m >= short.metrics.distance_m
    assert not any(grid.blocked[grid.cell_of(*p)] for p in safe.points)


# --- time estimation -------------------------------------------------------------------------------

def test_flat_ground_travel_time_is_distance_over_base_speed():
    grid = make_routing_grid(["." * 12] * 3)
    result = plan(grid, [stop_at(grid, 1, 0), stop_at(grid, 1, 10)])
    assert result.metrics.travel_time_s == pytest.approx(100.0 / CONFIG.speed.base_speed_mps)


def test_travel_time_slows_with_slope_from_the_path_geometry():
    flat = make_routing_grid(["." * 12] * 3, slope=0.0)
    at_limit = make_routing_grid(["." * 12] * 3, slope=15.0)
    half = make_routing_grid(["." * 12] * 3, slope=7.5)
    times = []
    for grid in (flat, half, at_limit):
        times.append(plan(grid, [stop_at(grid, 1, 0), stop_at(grid, 1, 10)]).metrics.travel_time_s)
    base = CONFIG.speed.base_speed_mps
    frac = CONFIG.speed.min_speed_fraction
    assert times[0] == pytest.approx(100.0 / base)
    assert times[1] == pytest.approx(100.0 / (base * (1 - (1 - frac) * 0.5)))
    assert times[2] == pytest.approx(100.0 / (base * frac))
    assert times[0] < times[1] < times[2]


def test_speed_model_is_configurable():
    fast = RoutingConfig(speed=SpeedModel(base_speed_mps=1.6))
    grid = make_routing_grid(["." * 12] * 3)
    result = plan_profile(grid, [stop_at(grid, 1, 0), stop_at(grid, 1, 10)], SHORT, fast)
    assert result.metrics.travel_time_s == pytest.approx(100.0 / 1.6)
    assert speed_mps(0.0, 15.0, fast.speed) == 1.6 and speed_mps(99.0, 15.0, fast.speed) == pytest.approx(1.6 * 0.4)


# --- multi-stop routes, waypoints, return leg ------------------------------------------------------

def test_multi_stop_route_with_activities_and_return_leg():
    grid = make_routing_grid(["." * 20] * 10)
    stops = [
        stop_at(grid, 0, 0, "habitat", kind=WaypointType.START),
        stop_at(grid, 0, 10, "target A", kind=WaypointType.SCIENCE_TARGET, activity_min=30),
        stop_at(grid, 9, 10, "waypoint", kind=WaypointType.WAYPOINT),
        stop_at(grid, 9, 19, "target B", kind=WaypointType.SCIENCE_TARGET, activity_min=45),
        stop_at(grid, 0, 0, "habitat", kind=WaypointType.HABITAT_RETURN),
    ]
    result = plan(grid, stops)
    m = result.metrics
    assert result.status == SearchStatus.FOUND and len(result.legs) == 4
    assert m.cumulative_distance_m[result.stop_vertices[1]] == pytest.approx(100.0)
    assert m.cumulative_distance_m[result.stop_vertices[2]] == pytest.approx(190.0)
    assert m.cumulative_distance_m[result.stop_vertices[3]] == pytest.approx(280.0)
    return_leg = m.distance_m - 280.0
    assert return_leg == pytest.approx(10.0 * (19 + (SQRT2 - 1) * 9))        # octile from (9,19) to (0,0)
    assert result.stop_vertices == sorted(result.stop_vertices) and len(set(result.stop_vertices)) == 5
    assert result.points[result.stop_vertices[-1]] == (stops[-1].x_m, stops[-1].y_m)
    arrivals = stop_arrival_times(result)
    speed = CONFIG.speed.base_speed_mps
    assert arrivals[1] == pytest.approx(100.0 / speed)
    assert arrivals[2] == pytest.approx(190.0 / speed + 30 * 60)             # after target A's activity
    assert arrivals[3] == pytest.approx(280.0 / speed + 30 * 60)
    assert arrivals[4] == pytest.approx(m.distance_m / speed + (30 + 45) * 60)
    assert activity_seconds(stops) == 75 * 60


def test_stops_are_visited_in_the_requested_order_even_when_a_reorder_would_be_shorter():
    grid = make_routing_grid(["." * 20] * 3)
    stops = [stop_at(grid, 1, 0), stop_at(grid, 1, 19, "far"), stop_at(grid, 1, 5, "near"), stop_at(grid, 1, 0)]
    result = plan(grid, stops)
    assert result.metrics.distance_m == pytest.approx(190.0 + 140.0 + 50.0)
    cols = [grid.cell_of(*result.points[v])[1] for v in result.stop_vertices]
    assert cols == [0, 19, 5, 0]


# --- search limits ---------------------------------------------------------------------------------

def test_expansion_limit_is_reported_as_a_limit_not_as_unreachable():
    grid = make_routing_grid(["." * 30] * 30)
    result = search(grid, (0, 0), (29, 29), SAFE, limits=SearchLimits(max_expansions=5))
    assert result.status == SearchStatus.LIMIT_EXCEEDED and result.cells == []


def test_timeout_is_reported_as_a_timeout():
    grid = make_routing_grid(["." * 30] * 30)
    ticks = iter(range(0, 10**6, 1000))
    result = search(grid, (0, 0), (29, 29), limits=SearchLimits(timeout_s=1.0), clock=lambda: float(next(ticks)))
    assert result.status == SearchStatus.TIMEOUT and result.expanded >= 1


def test_plan_stops_at_the_first_failed_leg_with_limit_status():
    grid = make_routing_grid(["." * 30] * 30)
    result = plan(grid, [stop_at(grid, 0, 0), stop_at(grid, 29, 29)], limits=SearchLimits(max_expansions=3))
    assert result.status == SearchStatus.LIMIT_EXCEEDED and result.metrics is None


def test_planner_requires_two_stops_and_in_grid_stops():
    grid = make_routing_grid(["...."] * 3)
    with pytest.raises(ValueError):
        plan(grid, [stop_at(grid, 0, 0)])
    far = Stop(WaypointType.WAYPOINT, "far", 1e6, 0.0)
    with pytest.raises(ValueError, match="outside"):
        plan(grid, [stop_at(grid, 0, 0), far])


# --- mission constraints applied to the grid ---------------------------------------------------------

def constraint_grid(slope, category=None, blocked=None, layer_limit=15.0, layer_blocks_unknown=True, constraints=None):
    base = make_routing_grid(["." * 10] * 6, slope=slope)
    return build_routing_grid(
        base.grid, base.elevation, base.slope_deg, base.risk,
        base.category if category is None else category,
        base.blocked if blocked is None else blocked,
        constraints or RoutingConstraints(15.0, True), layer_limit, layer_blocks_unknown, base.uncertainty,
    )


def test_tighter_mission_slope_limit_blocks_cells_and_changes_the_route():
    slope = np.zeros((6, 10), np.float32)
    slope[2:4, 3:7] = 11.0
    loose, loose_warn = constraint_grid(slope)
    tight, tight_warn = constraint_grid(slope, constraints=RoutingConstraints(10.0, True))
    assert not loose.blocked.any() and loose_warn == []
    assert tight.blocked[2:4, 3:7].all() and tight.slope_limit_deg == 10.0 and tight_warn == []
    stops = [stop_at(loose, 2, 0), stop_at(loose, 2, 9)]
    assert plan(loose, stops).metrics.distance_m == pytest.approx(90.0)
    assert plan(tight, stops).metrics.distance_m > 90.0


def test_slope_limit_can_make_a_destination_unreachable():
    slope = np.zeros((6, 10), np.float32)
    slope[:, 4] = 12.0                                                    # a steep ridge across the whole map
    tight, _ = constraint_grid(slope, constraints=RoutingConstraints(10.0, True))
    result = plan(tight, [stop_at(tight, 2, 0), stop_at(tight, 2, 9)])
    assert result.status == SearchStatus.UNREACHABLE


def test_mission_cannot_relax_the_processed_hazard_layer():
    slope = np.full((6, 10), 17.0, np.float32)
    grid, warnings = constraint_grid(slope, blocked=np.ones((6, 10), bool), constraints=RoutingConstraints(25.0, True))
    assert grid.slope_limit_deg == 15.0 and grid.blocked.all()
    assert any("exceeds the processed hazard layer" in w for w in warnings)


def test_unknown_terrain_blocking_follows_the_stricter_of_layer_and_mission():
    category = np.full((6, 10), 0, np.uint8)
    category[2, 2] = CATEGORY_UNKNOWN
    not_blocked = np.zeros((6, 10), bool)
    lenient_layer = dict(category=category, blocked=not_blocked, layer_blocks_unknown=False)
    strict_mission, _ = constraint_grid(0.0, constraints=RoutingConstraints(15.0, True), **lenient_layer)
    assert strict_mission.blocked[2, 2]
    lenient_mission, warn = constraint_grid(0.0, constraints=RoutingConstraints(15.0, False), **lenient_layer)
    assert not lenient_mission.blocked[2, 2] and warn == []
    strict_layer, warn = constraint_grid(0.0, category=category, blocked=not_blocked, layer_blocks_unknown=True,
                                         constraints=RoutingConstraints(15.0, False))
    assert strict_layer.blocked[2, 2] and any("stays blocked" in w for w in warn)


# --- the path itself is the source of truth --------------------------------------------------------

@pytest.mark.parametrize("profile", [SHORT, SAFE])
def test_every_vertex_and_segment_midpoint_lies_on_traversable_ground(profile):
    rng = np.random.default_rng(3)
    ascii_map = ["".join(rng.choice(list("....#~"), 25)) for _ in range(25)]
    ascii_map[0], ascii_map[-1] = "." * 25, "." * 25
    grid = make_routing_grid(ascii_map)
    result = plan(grid, [stop_at(grid, 0, 0), stop_at(grid, 24, 24)], profile)
    assert result.status == SearchStatus.FOUND
    xy = np.array(result.points)
    for (x0, y0), (x1, y1) in zip(xy, xy[1:]):
        for t in np.linspace(0, 1, 9):
            cell = grid.cell_of(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
            assert cell is not None and not grid.blocked[cell], f"segment crosses blocked cell {cell}"


def test_metrics_recomputed_from_geometry_match_what_the_planner_reported():
    grid = make_routing_grid(RISK_BAND, slope=np.linspace(0, 12, 15, dtype=np.float32)[None, :].repeat(9, 0))
    result = plan(grid, [stop_at(grid, 4, 0), stop_at(grid, 4, 14)], SAFE)
    again = evaluate_geometry(grid, result.points, SAFE, CONFIG)
    for field in ("distance_m", "travel_time_s", "total_cost", "risk_score", "peak_risk", "mean_slope_deg",
                  "max_slope_deg", "elevation_gain_m", "high_risk_distance_m", "uncertain_distance_m"):
        assert getattr(again, field) == pytest.approx(getattr(result.metrics, field))
    # risk metrics are bounded and consistent
    assert 0.0 <= result.metrics.risk_score <= result.metrics.peak_risk <= 1.0


def test_elevation_gain_and_loss_come_from_the_path():
    grid = make_routing_grid(["." * 11] * 3)
    elevation = np.zeros((3, 11), np.float32)
    elevation[:, :] = np.concatenate([np.arange(6), np.arange(4, -1, -1)]) * 2.0   # up 10 m then down 10 m
    grid = type(grid)(grid.grid, elevation, grid.slope_deg, grid.risk, grid.category, grid.blocked,
                      grid.uncertainty, grid.slope_limit_deg)
    result = plan(grid, [stop_at(grid, 1, 0), stop_at(grid, 1, 10)])
    assert result.metrics.elevation_gain_m == pytest.approx(10.0)
    assert result.metrics.elevation_loss_m == pytest.approx(10.0)
