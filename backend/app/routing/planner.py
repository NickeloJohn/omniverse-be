"""Multi-stop route planning and path evaluation.

Every reported metric (distance, time, risk, slope, elevation change, cost) is recomputed here from the final path
geometry, segment by segment, independently of the search. A segment's risk, slope and uncertainty are the mean of
the two cells at its ends, which is exactly what the A* step cost uses, so the metrics and the search agree.
"""

import math
from dataclasses import dataclass, field
from uuid import UUID

import numpy as np

from app.models.enums import RouteProfile, WaypointType
from app.processing.hazard import CATEGORY_HIGH
from app.routing.astar import SearchLimits, SearchResult, SearchSpace, SearchStatus, astar, validate_cell_path
from app.routing.cost_model import RoutingConfig, cell_factors, effective_slope, speed_mps
from app.routing.grid import RoutingGrid

EPS = 1e-9


@dataclass(frozen=True)
class Stop:
    kind: WaypointType
    label: str
    x_m: float
    y_m: float
    target_id: UUID | None = None
    activity_min: float = 0.0


@dataclass
class PathMetrics:
    distance_m: float
    travel_time_s: float
    total_cost: float
    risk_score: float
    peak_risk: float
    mean_slope_deg: float | None
    max_slope_deg: float | None
    elevation_gain_m: float
    elevation_loss_m: float
    high_risk_distance_m: float
    uncertain_distance_m: float
    cumulative_distance_m: np.ndarray   # per vertex
    cumulative_time_s: np.ndarray       # per vertex


@dataclass
class LegPlan:
    from_index: int
    to_index: int
    expanded: int
    status: SearchStatus


@dataclass
class ProfilePlan:
    profile: RouteProfile
    status: SearchStatus
    stops: list[Stop]
    failed_leg: int | None = None
    blocked_stop: int | None = None
    points: list[tuple[float, float]] = field(default_factory=list)
    stop_vertices: list[int] = field(default_factory=list)
    metrics: PathMetrics | None = None
    legs: list[LegPlan] = field(default_factory=list)
    expanded: int = 0


def _cell_values(grid: RoutingGrid, points: list[tuple[float, float]]) -> list[tuple[int, int]]:
    cells = []
    for x, y in points:
        cell = grid.cell_of(x, y)
        if cell is None:
            raise ValueError(f"path vertex {(x, y)} is outside the grid")
        cells.append(cell)
    return cells


def evaluate_geometry(
    grid: RoutingGrid, points: list[tuple[float, float]], profile: RouteProfile, config: RoutingConfig
) -> PathMetrics:
    """Compute all path metrics from the polyline `points` (local metres)."""
    cells = _cell_values(grid, points)
    rows = np.array([c[0] for c in cells])
    cols = np.array([c[1] for c in cells])
    xy = np.asarray(points, dtype=np.float64)
    seg_len = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))

    factors = cell_factors(grid, profile, config)[rows, cols]
    risk = grid.risk[rows, cols].astype(np.float64)
    slope = effective_slope(grid.slope_deg[rows, cols], grid.slope_limit_deg)
    uncertainty = grid.uncertainty[rows, cols].astype(np.float64)
    elevation = grid.elevation[rows, cols].astype(np.float64)
    high = grid.category[rows, cols] == CATEGORY_HIGH

    def mean_pair(values: np.ndarray) -> np.ndarray:
        return (values[:-1] + values[1:]) / 2.0

    seg_speed = speed_mps(mean_pair(slope), grid.slope_limit_deg, config.speed)
    seg_time = seg_len / seg_speed
    distance = float(seg_len.sum())
    if distance > 0:
        risk_score = float((seg_len * mean_pair(risk)).sum() / distance)
        mean_slope = float((seg_len * mean_pair(slope)).sum() / distance)
    else:
        risk_score, mean_slope = float(risk[0]), float(slope[0])

    known_slope = grid.slope_deg[rows, cols]
    finite_slope = known_slope[np.isfinite(known_slope)]
    delta = np.diff(elevation)
    delta = delta[np.isfinite(delta)]
    return PathMetrics(
        distance_m=distance,
        travel_time_s=float(seg_time.sum()),
        total_cost=float((seg_len * mean_pair(factors)).sum()),
        risk_score=risk_score,
        peak_risk=float(risk.max()),
        mean_slope_deg=mean_slope,
        max_slope_deg=float(finite_slope.max()) if finite_slope.size else None,
        elevation_gain_m=float(delta[delta > 0].sum()),
        elevation_loss_m=float(-delta[delta < 0].sum()),
        high_risk_distance_m=float(seg_len[high[:-1] | high[1:]].sum()),
        uncertain_distance_m=float(seg_len[mean_pair(uncertainty) > 0].sum()),
        cumulative_distance_m=np.concatenate([[0.0], np.cumsum(seg_len)]),
        cumulative_time_s=np.concatenate([[0.0], np.cumsum(seg_time)]),
    )


def _leg_points(
    grid: RoutingGrid, cells: list[tuple[int, int]], start: Stop, end: Stop
) -> list[tuple[float, float]]:
    """Exact start point, cell centres, exact end point; duplicates within EPS are dropped."""
    points = [(start.x_m, start.y_m)]
    for row, col in cells:
        centre = grid.grid.cell_center(row, col)
        if math.hypot(centre[0] - points[-1][0], centre[1] - points[-1][1]) > EPS:
            points.append(centre)
    if math.hypot(end.x_m - points[-1][0], end.y_m - points[-1][1]) > EPS:
        points.append((end.x_m, end.y_m))
    return points


def plan_profile(
    grid: RoutingGrid,
    stops: list[Stop],
    profile: RouteProfile,
    config: RoutingConfig,
    limits: SearchLimits = SearchLimits(),
    space: SearchSpace | None = None,
) -> ProfilePlan:
    """Plan start -> ... -> end through `stops` in the given order, one A* search per leg."""
    if len(stops) < 2:
        raise ValueError("at least two stops are required")
    space = space or SearchSpace(grid, cell_factors(grid, profile, config))
    plan = ProfilePlan(profile=profile, status=SearchStatus.FOUND, stops=stops)

    stop_cells = []
    for stop in stops:
        cell = grid.cell_of(stop.x_m, stop.y_m)
        if cell is None:
            raise ValueError(f"stop '{stop.label}' is outside the grid")
        stop_cells.append(cell)

    # Fail fast, before any search, if a stop sits on blocked terrain.
    for index, cell in enumerate(stop_cells):
        if grid.blocked[cell]:
            plan.status = SearchStatus.START_BLOCKED if index == 0 else SearchStatus.GOAL_BLOCKED
            plan.failed_leg, plan.blocked_stop = max(index - 1, 0), index
            return plan
    for index in range(len(stops) - 1):
        result: SearchResult = astar(space, stop_cells[index], stop_cells[index + 1], limits)
        plan.expanded += result.expanded
        plan.legs.append(LegPlan(index, index + 1, result.expanded, result.status))
        if result.status != SearchStatus.FOUND:
            plan.status, plan.failed_leg = result.status, index
            plan.points, plan.stop_vertices, plan.metrics = [], [], None
            return plan
        validate_cell_path(grid, result.cells)
        leg = _leg_points(grid, result.cells, stops[index], stops[index + 1])
        if not plan.points:
            plan.points = leg
            plan.stop_vertices = [0, len(leg) - 1]
        else:
            plan.points.extend(leg[1:])
            plan.stop_vertices.append(len(plan.points) - 1)

    plan.metrics = evaluate_geometry(grid, plan.points, profile, config)
    return plan


def activity_seconds(stops: list[Stop]) -> float:
    return sum(stop.activity_min for stop in stops) * 60.0


def stop_arrival_times(plan: ProfilePlan) -> list[float]:
    """Arrival time (s) at each stop: walking time so far plus activities completed at earlier stops."""
    assert plan.metrics is not None
    arrivals, activity = [], 0.0
    for stop, vertex in zip(plan.stops, plan.stop_vertices, strict=True):
        arrivals.append(float(plan.metrics.cumulative_time_s[vertex]) + activity)
        activity += stop.activity_min * 60.0
    return arrivals
