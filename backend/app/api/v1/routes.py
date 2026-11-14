from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.limits import route_planning_guard
from app.schemas.common import Page
from app.schemas.routes import MissionRouteRequest, PlannedRouteRead, RoutePlanRequest, RoutePlanResponse
from app.services import route_service
from app.services.storage import AssetStorage, get_storage

router = APIRouter(tags=["routes"])
DbSession = Annotated[Session, Depends(get_db)]
Storage = Annotated[AssetStorage, Depends(get_storage)]
PlannerGuard = Depends(route_planning_guard)


@router.post("/routes/plan", response_model=RoutePlanResponse, dependencies=[PlannerGuard])
def plan_route(payload: RoutePlanRequest, db: DbSession, storage: Storage) -> RoutePlanResponse:
    """Run A* over the processed terrain for each requested profile. Read-only; nothing is saved."""
    return route_service.plan_routes(db, storage, payload)


@router.get("/routes/{route_id}", response_model=PlannedRouteRead)
def get_route(route_id: UUID, db: DbSession) -> PlannedRouteRead:
    return route_service.route_to_read(db, route_service.get_route_or_404(db, route_id))


@router.post(
    "/missions/{mission_id}/routes",
    response_model=PlannedRouteRead,
    status_code=status.HTTP_201_CREATED,
    dependencies=[PlannerGuard],
)
def add_mission_route(
    mission_id: UUID, payload: MissionRouteRequest, db: DbSession, storage: Storage
) -> PlannedRouteRead:
    """Plan with the mission's context on the server and save the result as one of its routes."""
    return route_service.create_mission_route(db, storage, mission_id, payload)


@router.get("/missions/{mission_id}/routes", response_model=Page[PlannedRouteRead])
def list_mission_routes(
    mission_id: UUID,
    db: DbSession,
    limit: Annotated[int, Query(ge=1, le=100)] = 25,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Page[PlannedRouteRead]:
    items, total = route_service.list_mission_routes(db, mission_id, limit, offset)
    return Page[PlannedRouteRead](
        items=[route_service.route_to_read(db, r) for r in items], total=total, limit=limit, offset=offset
    )
