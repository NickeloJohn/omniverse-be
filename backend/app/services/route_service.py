"""Route planning orchestration: resolve inputs, run the A* engine on processed terrain, assemble and persist results.

Nothing here invents numbers. Distances, risk, slope, elevation change and times come from the path geometry
(routing.planner.evaluate_geometry); the time-budget check uses those computed times.
"""

import time
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import numpy as np
from geoalchemy2.shape import from_shape, to_shape
from shapely.geometry import LineString
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.errors import ConflictError, DomainValidationError, NotFoundError
from app.models import ExplorationSite, Mission, MissionTarget, Route, RouteWaypoint, ScienceTarget
from app.models.enums import DataOrigin, MissionStatus, RouteProfile, RouteStatus, WaypointType
from app.processing.hazard import DISCLAIMERS
from app.repositories import catalog, missions as mission_repo
from app.repositories.pagination import paginate
from app.routing.astar import SearchLimits, SearchSpace, SearchStatus
from app.routing.cost_model import (
    COST_FUNCTION_DESCRIPTIONS, DEFAULT_ROUTING_CONFIG, RoutingConfig, cell_factors, uncertainty_field,
)
from app.routing.grid import RoutingConstraints, RoutingGrid, build_routing_grid
from app.routing.planner import ProfilePlan, Stop, activity_seconds, plan_profile, stop_arrival_times
from app.schemas.mission import MissionConstraints
from app.schemas.routes import (
    ConstraintCheckRead, MissionRouteRequest, NoRouteRead, PlanTerrainRead, PlannedRouteRead, RouteComparisonRead,
    RouteGeometryRead, RouteLegRead, RouteMetricsRead, RoutePlanRequest, RoutePlanResponse, RouteStopRead,
    SearchStatsRead, PlanningInputs,
)
from app.services import hazard_service, terrain_service
from app.services.geo import ensure_within_site, geographic_for, point_geometry, point_xy
from app.services.storage import AssetStorage
from app.utils.mars_coords import LOCAL_SRID

DEFAULT_EVA_LIMIT_MIN, DEFAULT_RESERVE_MIN = 360, 60
MODEL_DISCLAIMER = (
    "Route risk, slope and time are prototype model outputs on orbital-resolution terrain. They are not a validated "
    "safety assessment or a prediction of astronaut or spacesuit performance."
)
FAILURE_CODES = {
    SearchStatus.START_BLOCKED: "start_blocked",
    SearchStatus.GOAL_BLOCKED: "destination_blocked",
    SearchStatus.UNREACHABLE: "unreachable",
    SearchStatus.LIMIT_EXCEEDED: "search_limit_exceeded",
    SearchStatus.TIMEOUT: "search_timeout",
}


@dataclass
class PlanContext:
    site: ExplorationSite
    mission: Mission | None
    stops: list[Stop]
    eva_limit_min: int
    reserve_min: int
    constraints: MissionConstraints
    return_planned: bool
    elevation_layer_id: UUID | None


def _mission_for(session: Session, mission_id: UUID | None) -> Mission | None:
    if mission_id is None:
        return None
    mission = mission_repo.get_mission(session, mission_id)
    if mission is None:
        raise NotFoundError("Mission not found")
    return mission


def _stop_from_target(
    target: ScienceTarget, label: str | None, duration: int | float | None, link: MissionTarget | None
) -> Stop:
    if duration is None:
        duration = link.planned_duration_min if link and link.planned_duration_min else target.estimated_activity_duration_min
    x, y = point_xy(target.location)
    return Stop(WaypointType.SCIENCE_TARGET, label or target.name, x, y, target.id, float(duration))


def resolve_context(session: Session, request: PlanningInputs, settings: Settings,
                    site_id: UUID | None = None, mission: Mission | None = None) -> PlanContext:
    if mission is not None:
        if site_id is not None and site_id != mission.site_id:
            raise DomainValidationError("site_id does not match the mission's site",
                                        [{"field": "site_id", "message": "mission belongs to a different site"}])
        site_id = mission.site_id
    site = catalog.get_site(session, site_id) if site_id else None
    if site is None:
        raise NotFoundError("Exploration site not found")

    habitat_x, habitat_y = point_xy(site.habitat_location)
    if request.start is not None:
        sx, sy = request.start.x_m, request.start.y_m
        ensure_within_site(site, sx, sy, "start")
    elif mission is not None:
        sx, sy = point_xy(mission.start_location)
    else:
        sx, sy = habitat_x, habitat_y
    stops = [Stop(WaypointType.START, "Start", sx, sy)]

    specs = request.destinations
    if specs is None and mission is not None:
        for link in mission.target_links:
            stops.append(_stop_from_target(link.target, None, None, link))
    elif specs:
        if len(specs) > settings.route_max_destinations:
            raise DomainValidationError(
                f"At most {settings.route_max_destinations} destinations can be planned at once",
                [{"field": "destinations", "message": f"{len(specs)} given"}],
            )
        links = {link.target_id: link for link in mission.target_links} if mission else {}
        waypoint_count = 0
        for index, spec in enumerate(specs):
            if spec.science_target_id is not None:
                target = mission_repo.get_science_target(session, spec.science_target_id)
                if target is None:
                    raise NotFoundError("Science target not found",
                                        [{"field": f"destinations[{index}]", "message": str(spec.science_target_id)}])
                if target.site_id != site.id:
                    raise DomainValidationError(
                        "Science target belongs to a different site",
                        [{"field": f"destinations[{index}]", "message": "target is not part of the site"}])
                stops.append(_stop_from_target(target, spec.label, spec.activity_duration_min, links.get(target.id)))
            else:
                assert spec.location is not None
                ensure_within_site(site, spec.location.x_m, spec.location.y_m, f"destinations[{index}].location")
                waypoint_count += 1
                stops.append(Stop(WaypointType.WAYPOINT, spec.label or f"Waypoint {waypoint_count}",
                                  spec.location.x_m, spec.location.y_m, None, float(spec.activity_duration_min or 0)))
    if len(stops) - 1 > settings.route_max_destinations:
        raise DomainValidationError(
            f"At most {settings.route_max_destinations} destinations can be planned at once",
            [{"field": "destinations", "message": f"{len(stops) - 1} resolved"}],
        )

    if request.return_to_habitat:
        stops.append(Stop(WaypointType.HABITAT_RETURN, "Habitat", habitat_x, habitat_y))
    if len(stops) < 2:
        raise DomainValidationError("Nothing to plan: add a destination or return to the habitat")

    limit = request.eva_duration_limit_min or (mission.eva_duration_limit_min if mission else DEFAULT_EVA_LIMIT_MIN)
    reserve = request.return_reserve_min if request.return_reserve_min is not None else (
        mission.return_reserve_min if mission else DEFAULT_RESERVE_MIN)
    if reserve >= limit:
        raise DomainValidationError("return_reserve_min must be less than eva_duration_limit_min",
                                    [{"field": "return_reserve_min", "message": f"{reserve} >= {limit}"}])
    if request.constraints is not None:
        constraints = request.constraints
    elif mission is not None:
        constraints = MissionConstraints.model_validate(mission.constraints or {})
    else:
        constraints = MissionConstraints()
    return PlanContext(site, mission, stops, limit, reserve, constraints, request.return_to_habitat,
                       request.elevation_layer_id)


def build_grid(session: Session, storage: AssetStorage, ctx: PlanContext, config: RoutingConfig
               ) -> tuple[RoutingGrid, list[str], terrain_service.TerrainData]:
    terrain = terrain_service.load_terrain(session, storage, ctx.site, ctx.elevation_layer_id)
    cost = hazard_service.get_cost_grid(session, terrain)
    unknown = ~np.isfinite(terrain.elevation)
    grid, warnings = build_routing_grid(
        terrain.grid, terrain.elevation, terrain.slope, cost.risk, cost.category, cost.blocked,
        RoutingConstraints(ctx.constraints.max_slope_deg, ctx.constraints.block_unknown_terrain),
        terrain.config.excluded_above_deg, terrain.config.block_unknown,
        uncertainty_field(unknown, terrain.grid.resolution_m, config.safety.uncertainty_buffer_m),
    )
    return grid, warnings, terrain


def _blocked_reason(grid: RoutingGrid, row: int, col: int) -> str:
    slope = grid.slope_deg[row, col]
    if not np.isfinite(grid.elevation[row, col]):
        return "unknown terrain (no elevation data)"
    if not np.isfinite(slope):
        return "unknown terrain (slope unavailable at borders or next to unknown data)"
    if slope > grid.slope_limit_deg:
        return f"slope {slope:.1f} deg exceeds the {grid.slope_limit_deg:g} deg limit"
    return "inside a blocking hazard zone or excluded terrain"


def _no_route(plan: ProfilePlan, grid: RoutingGrid) -> NoRouteRead:
    code = FAILURE_CODES[plan.status]
    leg = plan.failed_leg
    from_stop = plan.stops[leg] if leg is not None else None
    to_stop = plan.stops[leg + 1] if leg is not None else None
    if plan.blocked_stop is not None:
        stop = plan.stops[plan.blocked_stop]
        row, col = grid.cell_of(stop.x_m, stop.y_m)  # type: ignore[misc]
        message = f"'{stop.label}' is on blocked terrain: {_blocked_reason(grid, row, col)}"
    elif plan.status == SearchStatus.UNREACHABLE:
        message = f"No traversable path exists from '{from_stop.label}' to '{to_stop.label}'"  # type: ignore[union-attr]
    elif plan.status == SearchStatus.TIMEOUT:
        message = "The search ran out of time before finding a route; try fewer or closer stops"
    else:
        message = "The search exceeded its work limit before finding a route; try fewer or closer stops"
    return NoRouteRead(code=code, message=message, leg_index=leg,
                       from_label=from_stop.label if from_stop else None, to_label=to_stop.label if to_stop else None)


def _cost_model(profile: RouteProfile, config: RoutingConfig, grid: RoutingGrid, terrain) -> dict[str, Any]:
    model: dict[str, Any] = {
        "routing_config_version": config.version,
        "profile": profile.value,
        "formula": COST_FUNCTION_DESCRIPTIONS[profile],
        "slope_limit_deg_applied": grid.slope_limit_deg,
        "hazard_config_version": terrain.config.version,
        "speed_model": config.speed.model_dump(),
        "movement": "8-connected, no diagonal moves between blocked corners",
        "heuristic": "octile distance (admissible and consistent)",
        "disclaimer": MODEL_DISCLAIMER,
    }
    if profile == RouteProfile.SAFETY_FIRST:
        model["safety_weights"] = config.safety.model_dump()
    return model


def _assemble(plan: ProfilePlan, ctx: PlanContext, grid: RoutingGrid, terrain, config: RoutingConfig,
              elapsed_ms: float, extra_warnings: list[str]) -> PlannedRouteRead:
    site = ctx.site
    origin = DataOrigin.SIMULATED if terrain.is_demo else DataOrigin.DERIVED
    usable_s = (ctx.eva_limit_min - ctx.reserve_min) * 60.0
    warnings = list(extra_warnings)
    if not ctx.return_planned:
        warnings.append("Return to the habitat is not planned, so the return reserve cannot be verified")
    check = dict(
        eva_duration_limit_min=ctx.eva_limit_min, return_reserve_min=ctx.reserve_min, usable_time_s=usable_s,
        return_planned=ctx.return_planned, max_slope_deg_applied=grid.slope_limit_deg,
        block_unknown_terrain_applied=bool(ctx.constraints.block_unknown_terrain or terrain.config.block_unknown),
    )
    cost_model = _cost_model(plan.profile, config, grid, terrain)

    if plan.status != SearchStatus.FOUND or plan.metrics is None:
        return PlannedRouteRead(
            profile=plan.profile, status=RouteStatus.NO_ROUTE, no_route=_no_route(plan, grid), geometry=None,
            metrics=None, stops=[], legs=[],
            constraints=ConstraintCheckRead(**check, planned_total_time_s=0.0, margin_s=usable_s,
                                            within_time_budget=False, violations=["no_route"]),
            search=SearchStatsRead(expanded_nodes=plan.expanded, elapsed_ms=elapsed_ms),
            cost_model=cost_model, data_origin=origin, warnings=warnings,
        )

    m = plan.metrics
    activity_s = activity_seconds(plan.stops)
    total_s = m.travel_time_s + activity_s
    arrivals = stop_arrival_times(plan)
    stops, legs = [], []
    for index, (stop, vertex, arrival) in enumerate(zip(plan.stops, plan.stop_vertices, arrivals, strict=True)):
        row, col = grid.cell_of(stop.x_m, stop.y_m)  # type: ignore[misc]
        lon, lat = geographic_for(site, stop.x_m, stop.y_m)
        stops.append(RouteStopRead(
            sequence=index, waypoint_type=stop.kind, label=stop.label, location={"x_m": stop.x_m, "y_m": stop.y_m},
            longitude_deg_east=lon, latitude_deg=lat, science_target_id=stop.target_id,
            activity_duration_min=stop.activity_min, distance_from_start_m=float(m.cumulative_distance_m[vertex]),
            arrival_time_s=arrival, departure_time_s=arrival + stop.activity_min * 60.0,
            risk_score=float(grid.risk[row, col]),
        ))
        if index:
            previous = plan.stop_vertices[index - 1]
            legs.append(RouteLegRead(
                from_sequence=index - 1, to_sequence=index,
                distance_m=float(m.cumulative_distance_m[vertex] - m.cumulative_distance_m[previous]),
                travel_time_s=float(m.cumulative_time_s[vertex] - m.cumulative_time_s[previous]),
            ))

    violations = [] if total_s <= usable_s else ["eva_time_budget_exceeded"]
    if violations:
        warnings.append(
            f"Planned time {total_s / 60:.0f} min exceeds the usable EVA time {usable_s / 60:.0f} min "
            f"({ctx.eva_limit_min} min limit minus {ctx.reserve_min} min return reserve)"
        )
    lonlat = [geographic_for(site, x, y) for x, y in plan.points]
    return PlannedRouteRead(
        profile=plan.profile, status=RouteStatus.FOUND, no_route=None,
        geometry=RouteGeometryRead(point_count=len(plan.points), coordinates=plan.points, geographic=lonlat),
        metrics=RouteMetricsRead(
            distance_m=m.distance_m, travel_time_s=m.travel_time_s, activity_time_s=activity_s, total_time_s=total_s,
            total_cost=m.total_cost, risk_score=m.risk_score, peak_risk=m.peak_risk, mean_slope_deg=m.mean_slope_deg,
            max_slope_deg=m.max_slope_deg, elevation_gain_m=m.elevation_gain_m, elevation_loss_m=m.elevation_loss_m,
            high_risk_distance_m=m.high_risk_distance_m, uncertain_distance_m=m.uncertain_distance_m,
        ),
        stops=stops, legs=legs,
        constraints=ConstraintCheckRead(**check, planned_total_time_s=total_s, margin_s=usable_s - total_s,
                                        within_time_budget=not violations, violations=violations),
        search=SearchStatsRead(expanded_nodes=plan.expanded, elapsed_ms=elapsed_ms),
        cost_model=cost_model, data_origin=origin, warnings=warnings,
    )


def compare(routes: list[PlannedRouteRead]) -> RouteComparisonRead | None:
    by_profile = {r.profile: r for r in routes if r.metrics is not None and r.geometry is not None}
    short, safe = by_profile.get(RouteProfile.SHORTER_DISTANCE), by_profile.get(RouteProfile.SAFETY_FIRST)
    if short is None or safe is None:
        return None
    sm, fm = short.metrics, safe.metrics
    assert sm is not None and fm is not None and short.geometry and safe.geometry
    same = short.geometry.coordinates == safe.geometry.coordinates
    diff = fm.distance_m - sm.distance_m
    ratio = fm.distance_m / sm.distance_m if sm.distance_m > 0 else None
    if same:
        summary = "Both profiles produced the same path."
    else:
        summary = (
            f"Safety-first is {abs(diff):.0f} m {'longer' if diff >= 0 else 'shorter'} than the shortest route "
            f"and has a mean risk of {fm.risk_score:.3f} versus {sm.risk_score:.3f} "
            f"(peak {fm.peak_risk:.2f} versus {sm.peak_risk:.2f})."
        )
    return RouteComparisonRead(
        distance_difference_m=diff, distance_ratio=ratio, risk_score_difference=fm.risk_score - sm.risk_score,
        peak_risk_difference=fm.peak_risk - sm.peak_risk, travel_time_difference_s=fm.travel_time_s - sm.travel_time_s,
        same_geometry=same, summary=summary,
    )


def run_plan(session: Session, storage: AssetStorage, ctx: PlanContext, profiles: list[RouteProfile],
             settings: Settings | None = None, config: RoutingConfig = DEFAULT_ROUTING_CONFIG
             ) -> tuple[RoutePlanResponse, dict[RouteProfile, ProfilePlan], terrain_service.TerrainData]:
    settings = settings or get_settings()
    grid, grid_warnings, terrain = build_grid(session, storage, ctx, config)
    limits = SearchLimits(settings.route_max_expansions, settings.route_search_timeout_s)
    routes, plans = [], {}
    for profile in profiles:
        started = time.perf_counter()
        space = SearchSpace(grid, cell_factors(grid, profile, config))
        plan = plan_profile(grid, ctx.stops, profile, config, limits, space)
        elapsed = (time.perf_counter() - started) * 1000.0
        plans[profile] = plan
        routes.append(_assemble(plan, ctx, grid, terrain, config, round(elapsed, 2), grid_warnings))
    notes = [*terrain_service._notes(terrain.is_demo, []), MODEL_DISCLAIMER]
    response = RoutePlanResponse(
        site_id=ctx.site.id, mission_id=ctx.mission.id if ctx.mission else None,
        terrain=PlanTerrainRead(
            is_demo=terrain.is_demo, resolution_m=terrain.grid.resolution_m, elevation_layer=terrain.elevation_ref,
            slope_layer=terrain.slope_ref, hazard_layer=terrain.hazard_ref,
            hazard_config_version=terrain.config.version, routing_config_version=config.version,
        ),
        routes=routes, comparison=compare(routes), warnings=grid_warnings, notes=list(dict.fromkeys(notes)),
    )
    return response, plans, terrain


def plan_routes(session: Session, storage: AssetStorage, request: RoutePlanRequest,
                settings: Settings | None = None) -> RoutePlanResponse:
    """Compute routes for the requested profiles. Read-only: nothing is persisted."""
    settings = settings or get_settings()
    mission = _mission_for(session, request.mission_id)
    ctx = resolve_context(session, request, settings, request.site_id, mission)
    return run_plan(session, storage, ctx, request.profiles, settings)[0]


def create_mission_route(session: Session, storage: AssetStorage, mission_id: UUID, request: MissionRouteRequest,
                         settings: Settings | None = None) -> PlannedRouteRead:
    """Plan on the server and save the result as a mission route; client-supplied geometry is never accepted."""
    settings = settings or get_settings()
    mission = _mission_for(session, mission_id)
    assert mission is not None
    if mission.status not in {MissionStatus.DRAFT, MissionStatus.PLANNED}:
        raise ConflictError(f"Routes can only be added to draft or planned missions, not {mission.status.value}")
    ctx = resolve_context(session, request, settings, None, mission)
    response, plans, terrain = run_plan(session, storage, ctx, [request.profile], settings)
    planned = response.routes[0]
    if planned.status != RouteStatus.FOUND or planned.geometry is None or planned.metrics is None:
        info = planned.no_route
        raise ConflictError(
            f"No feasible route: {info.message if info else 'unknown'}",
            [{"profile": request.profile.value, "code": info.code if info else "no_route"}],
        )
    m = planned.metrics
    points = planned.geometry.coordinates
    route_grid, _, _ = build_grid(session, storage, ctx, DEFAULT_ROUTING_CONFIG)
    path_samples = []
    path_distance = 0.0
    for index, (x, y) in enumerate(points):
        if index:
            previous = points[index - 1]
            path_distance += float(np.hypot(x - previous[0], y - previous[1]))
        cell = route_grid.cell_of(x, y)
        if cell is None:
            raise DomainValidationError("Computed route vertex is outside the site grid")
        row, col = cell
        slope = float(route_grid.slope_deg[row, col])
        path_samples.append({
            "distance_m": path_distance,
            "risk_score": float(route_grid.risk[row, col]),
            "slope_deg": slope if np.isfinite(slope) else None,
        })
    route = Route(
        mission_id=mission.id, profile=request.profile, status=RouteStatus.FOUND, is_selected=request.select,
        geometry=from_shape(LineString(points if len(points) > 1 else [points[0], points[0]]), srid=LOCAL_SRID),
        distance_m=m.distance_m, estimated_duration_s=m.travel_time_s, total_cost=m.total_cost,
        risk_score=m.risk_score, max_cell_risk=m.peak_risk, mean_slope_deg=m.mean_slope_deg,
        max_slope_deg=m.max_slope_deg, elevation_gain_m=m.elevation_gain_m, elevation_loss_m=m.elevation_loss_m,
        terrain_layer_id=terrain.elevation_ref["id"], dataset_version=str(terrain.hazard_ref["dataset_version"])[:64],
        cost_model={
            "details": planned.model_dump(mode="json", exclude={"geometry", "id", "mission_id", "is_selected"}),
            "path_samples": path_samples,
        },
        data_origin=planned.data_origin,
    )
    route.waypoints = [
        RouteWaypoint(
            sequence=s.sequence, waypoint_type=s.waypoint_type, location=point_geometry(s.location.x_m, s.location.y_m),
            science_target_id=s.science_target_id, distance_from_start_m=s.distance_from_start_m,
            eta_s=s.arrival_time_s, risk_score=s.risk_score,
        )
        for s in planned.stops
    ]
    if request.select:
        session.execute(update(Route).where(Route.mission_id == mission.id).values(is_selected=False))
    session.add(route)
    session.commit()
    return route_to_read(session, route)


def route_to_read(session: Session, route: Route) -> PlannedRouteRead:
    """Rebuild the API view of a saved route from its stored geometry and stored computed details."""
    mission = session.get(Mission, route.mission_id)
    site = session.get(ExplorationSite, mission.site_id) if mission else None
    details = dict(route.cost_model.get("details", {}))
    geometry = None
    if route.geometry is not None and site is not None:
        coords = [(float(x), float(y)) for x, y in to_shape(route.geometry).coords]
        geometry = RouteGeometryRead(point_count=len(coords), coordinates=coords,
                                     geographic=[geographic_for(site, x, y) for x, y in coords])
    return PlannedRouteRead(**details, id=route.id, mission_id=route.mission_id, is_selected=route.is_selected,
                            geometry=geometry)


def get_route_or_404(session: Session, route_id: UUID) -> Route:
    route = session.get(Route, route_id)
    if route is None:
        raise NotFoundError("Route not found")
    return route


def list_mission_routes(session: Session, mission_id: UUID, limit: int, offset: int) -> tuple[list[Route], int]:
    if mission_repo.get_mission(session, mission_id) is None:
        raise NotFoundError("Mission not found")
    stmt = select(Route).where(Route.mission_id == mission_id).order_by(Route.created_at.desc(), Route.id)
    return paginate(session, stmt, limit, offset)
