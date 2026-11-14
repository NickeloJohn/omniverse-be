"""Deterministic, persistent EVA simulation driven only by explicit logical-clock advances.

Resource rates, risk thresholds and speed/time values are prototype model inputs, not astronaut or spacesuit
measurements. All session and telemetry data is constrained to data_origin='simulated'. No wall-clock time is used
to move the astronaut; POST /advance is the authoritative simulation clock.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import numpy as np
from geoalchemy2.shape import from_shape, to_shape
from shapely.geometry import Point
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session, selectinload

from app.core.errors import ConflictError, DomainValidationError, NotFoundError
from app.models import EVASession, Mission, Route, TelemetrySample
from app.models.enums import DataOrigin, EVAStatus, MissionStatus, RiskLevel, RouteStatus
from app.models.base import utcnow
from app.schemas.simulation import (
    ResourceModel,
    SimulationAdvanceRead,
    SimulationStateRead,
    SimulationStartedRead,
    SimulationStartRequest,
    SimulationResetRead,
    TelemetryRead,
)
from app.services.geo import point_geometry
from app.utils.mars_coords import LOCAL_SRID


@dataclass(frozen=True)
class RouteSchedule:
    coordinates: list[tuple[float, float]]
    cumulative_distance_m: np.ndarray
    stops: list[dict[str, Any]]
    total_distance_m: float
    total_time_s: float
    travel_time_s: float
    activity_time_s: float
    has_return: bool
    risk_samples: list[dict[str, Any]]


def _get_session(session: Session, session_id: UUID, lock: bool = False) -> EVASession:
    stmt = select(EVASession).options(selectinload(EVASession.telemetry)).where(EVASession.id == session_id)
    if lock:
        stmt = stmt.with_for_update()
    eva = session.scalars(stmt).unique().first()
    if eva is None:
        raise NotFoundError("Simulation not found")
    return eva


def _get_route(session: Session, route_id: UUID) -> Route:
    route = session.get(Route, route_id)
    if route is None:
        raise NotFoundError("Route not found")
    return route


def _schedule(route: Route) -> RouteSchedule:
    if route.status != RouteStatus.FOUND or route.geometry is None or route.distance_m is None:
        raise DomainValidationError("Simulation requires a found route with saved geometry")
    details = route.cost_model.get("details", {})
    stops = details.get("stops") or []
    metrics = details.get("metrics") or {}
    if len(stops) < 2 or not metrics:
        raise DomainValidationError("Route is missing its computed schedule; save a newly planned route")
    coordinates = [(float(x), float(y)) for x, y in to_shape(route.geometry).coords]
    if not coordinates:
        raise DomainValidationError("Route geometry is empty")
    points = np.asarray(coordinates, dtype=np.float64)
    segment_lengths = np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1]))
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_time = float(metrics["total_time_s"])
    if not math.isfinite(total_time) or total_time < 0:
        raise DomainValidationError("Route has an invalid computed duration")
    has_return = stops[-1].get("waypoint_type") == "habitat_return"
    return RouteSchedule(
        coordinates=coordinates,
        cumulative_distance_m=cumulative,
        stops=stops,
        total_distance_m=float(route.distance_m),
        total_time_s=total_time,
        travel_time_s=float(metrics["travel_time_s"]),
        activity_time_s=float(metrics["activity_time_s"]),
        has_return=has_return,
        risk_samples=route.cost_model.get("path_samples", []),
    )


def _position_at_distance(schedule: RouteSchedule, distance_m: float) -> tuple[float, float]:
    cumulative = schedule.cumulative_distance_m
    if distance_m <= 0 or len(cumulative) == 1:
        return schedule.coordinates[0]
    if distance_m >= cumulative[-1]:
        return schedule.coordinates[-1]
    index = int(np.searchsorted(cumulative, distance_m, side="right") - 1)
    index = min(index, len(schedule.coordinates) - 2)
    span = float(cumulative[index + 1] - cumulative[index])
    if span <= 1e-12:
        return schedule.coordinates[index + 1]
    fraction = (distance_m - float(cumulative[index])) / span
    x0, y0 = schedule.coordinates[index]
    x1, y1 = schedule.coordinates[index + 1]
    return x0 + fraction * (x1 - x0), y0 + fraction * (y1 - y0)


def _position_and_distance(schedule: RouteSchedule, sim_time_s: float) -> tuple[tuple[float, float], float]:
    """Follow saved stop arrival/departure times, holding position while a science activity is in progress."""
    stops = schedule.stops
    t = min(max(sim_time_s, 0.0), schedule.total_time_s)
    for index, stop in enumerate(stops):
        arrival = float(stop["arrival_time_s"])
        departure = float(stop["departure_time_s"])
        distance = float(stop["distance_from_start_m"])
        if t <= departure or index == len(stops) - 1:
            return _position_at_distance(schedule, distance), distance
        next_stop = stops[index + 1]
        next_arrival = float(next_stop["arrival_time_s"])
        next_distance = float(next_stop["distance_from_start_m"])
        if t < next_arrival:
            duration = next_arrival - departure
            fraction = 1.0 if duration <= 0 else min(max((t - departure) / duration, 0.0), 1.0)
            at_distance = distance + (next_distance - distance) * fraction
            return _position_at_distance(schedule, at_distance), at_distance
    return schedule.coordinates[-1], schedule.total_distance_m


def _time_components(schedule: RouteSchedule, sim_time_s: float) -> tuple[float, float]:
    """Return cumulative moving and activity seconds at logical time t, independent of advance chunking."""
    t = min(max(sim_time_s, 0.0), schedule.total_time_s)
    moving = activity = 0.0
    for index, stop in enumerate(schedule.stops):
        arrival = float(stop["arrival_time_s"])
        departure = float(stop["departure_time_s"])
        activity += max(0.0, min(t, departure) - min(t, arrival))
        if index + 1 < len(schedule.stops):
            leg_start = departure
            leg_end = float(schedule.stops[index + 1]["arrival_time_s"])
            moving += max(0.0, min(t, leg_end) - min(t, leg_start))
    return moving, activity


def _risk_at(schedule: RouteSchedule, distance_m: float) -> float | None:
    samples = schedule.risk_samples
    if not samples:
        return None
    distances = [float(sample["distance_m"]) for sample in samples]
    if distance_m <= distances[0]:
        return float(samples[0]["risk_score"])
    if distance_m >= distances[-1]:
        return float(samples[-1]["risk_score"])
    hi = int(np.searchsorted(distances, distance_m, side="right"))
    lo = hi - 1
    span = distances[hi] - distances[lo]
    fraction = (distance_m - distances[lo]) / span if span > 1e-12 else 0.0
    return float(samples[lo]["risk_score"]) * (1 - fraction) + float(samples[hi]["risk_score"]) * fraction


def _risk_level(risk: float | None, model: ResourceModel) -> RiskLevel:
    if risk is None:
        return RiskLevel.LOW
    if risk >= model.risk_critical_threshold:
        return RiskLevel.CRITICAL
    if risk >= model.risk_warning_threshold:
        return RiskLevel.HIGH
    if risk >= model.risk_warning_threshold * 0.75:
        return RiskLevel.ELEVATED
    return RiskLevel.LOW


def _return_time(schedule: RouteSchedule, sim_time_s: float, distance_m: float) -> tuple[float | None, str | None]:
    if schedule.has_return:
        return max(0.0, schedule.total_time_s - min(sim_time_s, schedule.total_time_s)), None
    if schedule.total_distance_m <= 1e-9 or schedule.travel_time_s <= 0:
        return 0.0, "Return time assumes retracing the traveled route; the planned route has no habitat return leg."
    return distance_m * schedule.travel_time_s / schedule.total_distance_m, (
        "Return time is an estimate based on retracing the traveled route; the planned route has no habitat return leg."
    )


def _resource_state(schedule: RouteSchedule, sim_time_s: float, model: ResourceModel) -> tuple[float, float]:
    moving_s, activity_s = _time_components(schedule, sim_time_s)
    oxygen = 100.0 - (
        model.oxygen_use_pct_per_hour_moving * moving_s
        + model.oxygen_use_pct_per_hour_activity * activity_s
    ) / 3600.0
    battery = 100.0 - (
        model.battery_use_pct_per_hour_moving * moving_s
        + model.battery_use_pct_per_hour_activity * activity_s
    ) / 3600.0
    return max(0.0, min(100.0, oxygen)), max(0.0, min(100.0, battery))


def _warnings(
    mission: Mission, schedule: RouteSchedule, sim_time_s: float, distance_m: float,
    oxygen: float, battery: float, risk: float | None, model: ResourceModel,
) -> tuple[list[str], float | None]:
    warnings: list[str] = []
    return_s, return_note = _return_time(schedule, sim_time_s, distance_m)
    if return_note:
        warnings.append(return_note)
    if oxygen <= model.oxygen_warning_pct:
        warnings.append(f"Simulated oxygen reserve is low ({oxygen:.1f}%)")
    if battery <= model.battery_warning_pct:
        warnings.append(f"Simulated battery reserve is low ({battery:.1f}%)")
    if risk is None:
        warnings.append("No saved per-cell route risk trace is available for this route")
    elif risk >= model.risk_critical_threshold:
        warnings.append(f"Critical planned terrain risk at current position ({risk:.2f})")
    elif risk >= model.risk_warning_threshold:
        warnings.append(f"Elevated planned terrain risk at current position ({risk:.2f})")

    eva_limit_s = mission.eva_duration_limit_min * 60.0
    remaining_eva_s = max(0.0, eva_limit_s - sim_time_s)
    if return_s is not None and return_s + mission.return_reserve_min * 60.0 > remaining_eva_s:
        warnings.append(
            f"Return-time check: estimated return ({return_s / 60:.1f} min) plus reserve "
            f"({mission.return_reserve_min} min) exceeds remaining EVA time ({remaining_eva_s / 60:.1f} min)"
        )
    if return_s is not None:
        oxygen_for_return = model.oxygen_use_pct_per_hour_moving * return_s / 3600.0
        battery_for_return = model.battery_use_pct_per_hour_moving * return_s / 3600.0
        if oxygen < oxygen_for_return:
            warnings.append(f"Simulated oxygen may be insufficient for estimated return ({oxygen_for_return:.1f}% required)")
        if battery < battery_for_return:
            warnings.append(f"Simulated battery may be insufficient for estimated return ({battery_for_return:.1f}% required)")
    return list(dict.fromkeys(warnings)), return_s


def _snapshot(session: Session, eva: EVASession) -> SimulationStateRead:
    route = _get_route(session, eva.route_id)
    mission = session.get(Mission, eva.mission_id)
    if mission is None:
        raise NotFoundError("Mission not found")
    schedule = _schedule(route)
    model = ResourceModel.model_validate(eva.resource_model)
    position, distance = _position_and_distance(schedule, eva.sim_elapsed_s)
    oxygen, battery = _resource_state(schedule, eva.sim_elapsed_s, model)
    risk = _risk_at(schedule, distance)
    warnings, return_s = _warnings(mission, schedule, eva.sim_elapsed_s, distance, oxygen, battery, risk, model)
    latest = session.scalars(
        select(TelemetrySample)
        .where(TelemetrySample.session_id == eva.id)
        .order_by(TelemetrySample.sim_time_s.desc())
        .limit(1)
    ).first()
    telemetry_count = session.scalar(
        select(func.count()).where(TelemetrySample.session_id == eva.id)
    ) or 0
    latest_read = TelemetryRead.model_validate({
        **{key: getattr(latest, key) for key in (
            "id", "session_id", "sim_time_s", "distance_along_route_m", "speed_mps", "oxygen_remaining_pct",
            "battery_remaining_pct", "estimated_return_time_s", "warnings", "data_origin", "created_at",
        )},
        "position": {"x_m": to_shape(latest.position).x, "y_m": to_shape(latest.position).y},
        "risk_level": latest.risk_level,
    }) if latest else None
    return SimulationStateRead(
        id=eva.id, mission_id=eva.mission_id, route_id=eva.route_id, status=eva.status,
        sim_elapsed_s=eva.sim_elapsed_s, distance_along_route_m=distance,
        total_route_distance_m=schedule.total_distance_m,
        route_progress_pct=100.0 if schedule.total_distance_m <= 0 else min(100.0, distance / schedule.total_distance_m * 100),
        position={"x_m": position[0], "y_m": position[1]},
        oxygen_remaining_pct=oxygen, battery_remaining_pct=battery, estimated_return_time_s=return_s,
        current_risk_score=risk, risk_level=_risk_level(risk, model), warnings=warnings,
        resource_model=model, telemetry_count=telemetry_count, latest_telemetry=latest_read,
        data_origin=DataOrigin.SIMULATED, created_at=eva.created_at, updated_at=eva.updated_at,
    )


def _record_telemetry(
    session: Session, eva: EVASession, mission: Mission, route: Route, sim_time_s: float,
    previous_distance_m: float,
) -> TelemetrySample:
    schedule = _schedule(route)
    position, distance = _position_and_distance(schedule, sim_time_s)
    oxygen, battery = _resource_state(schedule, sim_time_s, ResourceModel.model_validate(eva.resource_model))
    risk = _risk_at(schedule, distance)
    model = ResourceModel.model_validate(eva.resource_model)
    warnings, return_s = _warnings(mission, schedule, sim_time_s, distance, oxygen, battery, risk, model)
    elapsed_delta = sim_time_s - eva.sim_elapsed_s
    speed = max(0.0, distance - previous_distance_m) / elapsed_delta if elapsed_delta > 0 else 0.0
    sample = TelemetrySample(
        session_id=eva.id,
        sim_time_s=sim_time_s,
        position=from_shape(Point(*position), srid=LOCAL_SRID),
        distance_along_route_m=distance,
        speed_mps=speed,
        oxygen_remaining_pct=oxygen,
        battery_remaining_pct=battery,
        estimated_return_time_s=return_s,
        risk_level=_risk_level(risk, model).value,
        warnings=warnings,
        data_origin=DataOrigin.SIMULATED,
    )
    session.add(sample)
    return sample


def start_simulation(session: Session, payload: SimulationStartRequest) -> SimulationStartedRead:
    route = _get_route(session, payload.route_id)
    schedule = _schedule(route)
    mission = session.scalars(
        select(Mission).where(Mission.id == route.mission_id).with_for_update()
    ).first()
    if mission is None:
        raise NotFoundError("Mission not found")
    if mission.status not in {MissionStatus.DRAFT, MissionStatus.PLANNED}:
        raise ConflictError(f"Mission must be draft or planned to start an EVA, not {mission.status.value}")
    existing = session.scalars(select(EVASession).where(
        EVASession.mission_id == mission.id,
        EVASession.status.in_([EVAStatus.RUNNING, EVAStatus.PAUSED]),
    ).with_for_update()).first()
    if existing is not None:
        raise ConflictError("This mission already has a running or paused EVA session")

    eva = EVASession(
        mission_id=mission.id,
        route_id=route.id,
        status=EVAStatus.RUNNING,
        sim_elapsed_s=0.0,
        distance_along_route_m=0.0,
        oxygen_remaining_pct=100.0,
        battery_remaining_pct=100.0,
        resource_model=payload.resource_model.model_dump(mode="json"),
        started_at=utcnow(),
        data_origin=DataOrigin.SIMULATED,
    )
    mission.status = MissionStatus.ACTIVE
    session.add(eva)
    session.flush()
    _record_telemetry(session, eva, mission, route, 0.0, 0.0)
    session.commit()
    return SimulationStartedRead(
        session=_snapshot(session, eva), mission_status=mission.status.value,
        message="Simulation started. All position and resource values are simulated; advance_by_s is the authoritative clock.",
    )


def get_state(session: Session, session_id: UUID) -> SimulationStateRead:
    return _snapshot(session, _get_session(session, session_id))


def advance_simulation(session: Session, session_id: UUID, advance_by_s: float) -> SimulationAdvanceRead:
    eva = _get_session(session, session_id, lock=True)
    if eva.status != EVAStatus.RUNNING:
        raise ConflictError(f"Cannot advance a {eva.status.value} simulation; resume it first")
    route = _get_route(session, eva.route_id)
    schedule = _schedule(route)
    mission = session.get(Mission, eva.mission_id)
    if mission is None:
        raise NotFoundError("Mission not found")
    previous_distance = eva.distance_along_route_m
    previous_time = eva.sim_elapsed_s
    next_time = min(previous_time + advance_by_s, schedule.total_time_s)
    if next_time <= eva.sim_elapsed_s:
        eva.status = EVAStatus.COMPLETED
        mission.status = MissionStatus.COMPLETED
    else:
        _record_telemetry(session, eva, mission, route, next_time, previous_distance)
        eva.sim_elapsed_s = next_time
        _, distance = _position_and_distance(schedule, next_time)
        eva.distance_along_route_m = distance
        eva.oxygen_remaining_pct, eva.battery_remaining_pct = _resource_state(
            schedule, next_time, ResourceModel.model_validate(eva.resource_model)
        )
        if next_time >= schedule.total_time_s:
            eva.status = EVAStatus.COMPLETED
            mission.status = MissionStatus.COMPLETED
    session.commit()
    eva = _get_session(session, session_id)
    state = _snapshot(session, eva)
    latest = state.latest_telemetry
    if latest is None:
        raise RuntimeError("Simulation telemetry was not recorded")
    return SimulationAdvanceRead(
        session=state,
        telemetry=latest,
        advanced_by_s=max(0.0, next_time - previous_time),
        completed=eva.status == EVAStatus.COMPLETED,
    )


def pause_simulation(session: Session, session_id: UUID) -> SimulationStateRead:
    eva = _get_session(session, session_id, lock=True)
    if eva.status != EVAStatus.RUNNING:
        raise ConflictError(f"Cannot pause a {eva.status.value} simulation")
    mission = session.get(Mission, eva.mission_id)
    if mission is None:
        raise NotFoundError("Mission not found")
    eva.status = EVAStatus.PAUSED
    mission.status = MissionStatus.PAUSED
    session.commit()
    return _snapshot(session, _get_session(session, session_id))


def resume_simulation(session: Session, session_id: UUID) -> SimulationStateRead:
    eva = _get_session(session, session_id, lock=True)
    if eva.status != EVAStatus.PAUSED:
        raise ConflictError(f"Cannot resume a {eva.status.value} simulation")
    mission = session.get(Mission, eva.mission_id)
    if mission is None:
        raise NotFoundError("Mission not found")
    if mission.status != MissionStatus.PAUSED:
        raise ConflictError(f"Mission must be paused to resume the simulation, not {mission.status.value}")
    eva.status = EVAStatus.RUNNING
    mission.status = MissionStatus.ACTIVE
    session.commit()
    return _snapshot(session, _get_session(session, session_id))


def reset_simulation(session: Session, session_id: UUID) -> SimulationResetRead:
    eva = _get_session(session, session_id, lock=True)
    if eva.status == EVAStatus.ABORTED:
        raise ConflictError("An aborted simulation cannot be reset")
    mission = session.get(Mission, eva.mission_id)
    if mission is None:
        raise NotFoundError("Mission not found")
    route = _get_route(session, eva.route_id)
    eva.status = EVAStatus.CREATED
    eva.sim_elapsed_s = 0.0
    eva.distance_along_route_m = 0.0
    eva.oxygen_remaining_pct = 100.0
    eva.battery_remaining_pct = 100.0
    eva.started_at = None
    if mission.status in {MissionStatus.ACTIVE, MissionStatus.PAUSED, MissionStatus.COMPLETED}:
        mission.status = MissionStatus.PLANNED
    session.execute(delete(TelemetrySample).where(TelemetrySample.session_id == eva.id))
    session.flush()
    _record_telemetry(session, eva, mission, route, 0.0, 0.0)
    session.commit()
    eva = _get_session(session, session_id)
    return SimulationResetRead(session=_snapshot(session, eva), message="Simulation reset to its initial route position.",
                               data_origin=DataOrigin.SIMULATED)


def telemetry_page(session: Session, session_id: UUID, limit: int, offset: int) -> tuple[list[TelemetrySample], int]:
    eva = session.get(EVASession, session_id)
    if eva is None:
        raise NotFoundError("Simulation not found")
    stmt = select(TelemetrySample).where(TelemetrySample.session_id == session_id).order_by(TelemetrySample.sim_time_s)
    total = session.scalar(select(__import__("sqlalchemy").func.count()).select_from(stmt.subquery())) or 0
    return list(session.scalars(stmt.limit(limit).offset(offset))), total
