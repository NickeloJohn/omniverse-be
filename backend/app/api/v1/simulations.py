from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query, status
from geoalchemy2.shape import to_shape
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.common import Page
from app.schemas.simulation import (
    SimulationAdvanceRead,
    SimulationAdvanceRequest,
    SimulationResetRead,
    SimulationStartedRead,
    SimulationStartRequest,
    SimulationStateRead,
    TelemetryRead,
)
from app.services import simulation_service

router = APIRouter(prefix="/simulations", tags=["simulations"])
DbSession = Annotated[Session, Depends(get_db)]


@router.post("", response_model=SimulationStartedRead, status_code=status.HTTP_201_CREATED)
def start_simulation(payload: SimulationStartRequest, db: DbSession) -> SimulationStartedRead:
    """Start a deterministic simulation for a saved route; the logical clock advances only via POST /advance."""
    return simulation_service.start_simulation(db, payload)


@router.get("/{simulation_id}/state", response_model=SimulationStateRead)
def get_simulation_state(simulation_id: UUID, db: DbSession) -> SimulationStateRead:
    return simulation_service.get_state(db, simulation_id)


@router.get("/{simulation_id}/telemetry", response_model=Page[TelemetryRead])
def list_telemetry(
    simulation_id: UUID,
    db: DbSession,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Page[TelemetryRead]:
    items, total = simulation_service.telemetry_page(db, simulation_id, limit, offset)
    return Page[TelemetryRead](
        items=[TelemetryRead.model_validate({
            **{key: getattr(item, key) for key in (
                "id", "session_id", "sim_time_s", "distance_along_route_m", "speed_mps", "oxygen_remaining_pct",
                "battery_remaining_pct", "estimated_return_time_s", "risk_level", "warnings", "data_origin", "created_at",
            )},
            "position": {"x_m": to_shape(item.position).x, "y_m": to_shape(item.position).y},
        }) for item in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.post("/{simulation_id}/advance", response_model=SimulationAdvanceRead)
def advance_simulation(
    simulation_id: UUID, payload: SimulationAdvanceRequest, db: DbSession
) -> SimulationAdvanceRead:
    return simulation_service.advance_simulation(db, simulation_id, payload.advance_by_s)


@router.post("/{simulation_id}/pause", response_model=SimulationStateRead)
def pause_simulation(simulation_id: UUID, db: DbSession) -> SimulationStateRead:
    return simulation_service.pause_simulation(db, simulation_id)


@router.post("/{simulation_id}/resume", response_model=SimulationStateRead)
def resume_simulation(simulation_id: UUID, db: DbSession) -> SimulationStateRead:
    return simulation_service.resume_simulation(db, simulation_id)


@router.post("/{simulation_id}/reset", response_model=SimulationResetRead)
def reset_simulation(simulation_id: UUID, db: DbSession) -> SimulationResetRead:
    return simulation_service.reset_simulation(db, simulation_id)
